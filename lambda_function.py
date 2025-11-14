import json
import boto3
import numpy as np
from pyproj import CRS, Transformer
from io import StringIO


# Initialize S3 client once per container reuse.
s3 = boto3.client('s3')


# --------------------------------------------------------------------------- #
# Data loading utilities
# --------------------------------------------------------------------------- #
def read_trajectory_from_s3(bucket: str, key: str):
    """
    Read a Metashape trajectory file that contains position, orientation, and
    optionally the camera-to-world rotation matrix (r11–r33).
    """
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
    except Exception as exc:
        print(f"Error reading trajectory file from S3: {exc}")
        raise

    trajectories = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith('#') or not stripped:
            continue  # skip comments and blanks

        parts = stripped.split('\t')
        if len(parts) < 10:
            continue  # not enough data for pose

        try:
            photo_id = parts[0]
            x, y, z = map(float, parts[1:4])
            omega, phi, kappa = map(float, parts[4:7])

            if len(parts) >= 16:
                rotation_matrix = np.array(list(map(float, parts[7:16])), dtype=float).reshape(3, 3)
            else:
                rotation_matrix = euler_angles_to_matrix(omega, phi, kappa)

            trajectories.append(
                {
                    'photo_id': photo_id,
                    'x': x,
                    'y': y,
                    'z': z,
                    'omega': omega,
                    'phi': phi,
                    'kappa': kappa,
                    'rotation_matrix': rotation_matrix,
                }
            )
        except (ValueError, IndexError) as exc:
            print(f"Warning: skipping invalid line '{stripped}': {exc}")
            continue

    return trajectories


def euler_angles_to_matrix(omega: float, phi: float, kappa: float) -> np.ndarray:
    """Return the Metashape camera rotation matrix derived from OPK angles."""
    omega_rad = np.radians(omega)
    phi_rad = np.radians(phi)
    kappa_rad = np.radians(kappa)

    sin_o, cos_o = np.sin(omega_rad), np.cos(omega_rad)
    sin_p, cos_p = np.sin(phi_rad), np.cos(phi_rad)
    sin_k, cos_k = np.sin(kappa_rad), np.cos(kappa_rad)

    return np.array(
        [
            [cos_p * cos_k, cos_o * sin_k + sin_o * sin_p * cos_k, sin_o * sin_k - cos_o * sin_p * cos_k],
            [-cos_p * sin_k, cos_o * cos_k - sin_o * sin_p * sin_k, sin_o * cos_k + cos_o * sin_p * sin_k],
            [sin_p, -sin_o * cos_p, cos_o * cos_p],
        ],
        dtype=float,
    )


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def build_local_to_world_axes(affine_params):
    """
    Construct an orthonormal basis that maps Metashape local axes to world ENU.

    The affine fit already captures scaling and rotation in the horizontal
    plane; this builds an orthonormal basis we can apply to the rotation matrix
    so position and orientation use the same transform.
    """
    a, b, _, d, e, _ = affine_params

    local_x = np.array([a, d, 0.0], dtype=float)
    local_z = np.array([b, e, 0.0], dtype=float)

    x_norm = np.linalg.norm(local_x)
    if x_norm < 1e-8:
        raise ValueError("Affine transform produced a degenerate X axis.")
    x_unit = local_x / x_norm

    z_proj = local_z - np.dot(local_z, x_unit) * x_unit
    z_norm = np.linalg.norm(z_proj)
    if z_norm < 1e-8:
        z_proj = np.array([-x_unit[1], x_unit[0], 0.0], dtype=float)
        z_norm = np.linalg.norm(z_proj)
        if z_norm < 1e-8:
            raise ValueError("Affine transform cannot derive a stable Z axis.")
    z_unit = z_proj / z_norm

    y_unit = np.cross(x_unit, z_unit)
    y_norm = np.linalg.norm(y_unit)
    if y_norm < 1e-8:
        raise ValueError("Failed to compute vertical axis from affine transform.")
    y_unit = y_unit / y_norm

    if y_unit[2] < 0:
        y_unit = -y_unit
        z_unit = -z_unit

    return np.column_stack((x_unit, -y_unit, z_unit))


def wrap_to_180(angle_degrees: float) -> float:
    """Normalize an angle to [-180, 180)."""
    return (angle_degrees + 180.0) % 360.0 - 180.0


def clamp_index(trajectories, idx: int) -> int:
    """Clamp the manual reference index to the available trajectory range."""
    if not trajectories:
        raise ValueError("No trajectory data available.")
    return max(0, min(len(trajectories) - 1, idx))


def calculate_affine_params(manual_refs: dict, trajectories: list):
    """Solve the 2D affine transform that aligns Metashape X/Z to world EN coordinates."""
    if len(manual_refs) < 3:
        raise ValueError("At least 3 reference points are required for an affine transformation.")

    first_ref_idx = min(manual_refs.keys(), key=lambda key: int(key))
    first_lat, first_lon, _ = manual_refs[first_ref_idx]

    utm_zone = int((first_lon + 180) / 6) + 1
    hemisphere = 'north' if first_lat >= 0 else 'south'
    wgs84_crs = CRS.from_epsg(4326)
    utm_crs = CRS.from_dict(
        {
            'proj': 'utm',
            'zone': utm_zone,
            'hemisphere': hemisphere,
            'ellps': 'WGS84',
            'datum': 'WGS84',
            'units': 'm',
        }
    )
    transformer = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)

    local_points = []
    world_points = []

    for idx_str, (lat, lon, _) in manual_refs.items():
        idx = clamp_index(trajectories, int(idx_str))
        trajectory = trajectories[idx]

        local_points.append([trajectory['x'], trajectory['z']])
        utm_e, utm_n = transformer.transform(lon, lat)
        world_points.append([utm_e, utm_n])

    local_pts_np = np.asarray(local_points, dtype=float)
    world_pts_np = np.asarray(world_points, dtype=float)
    design_matrix = np.hstack([local_pts_np, np.ones((local_pts_np.shape[0], 1))])

    params_x, _, _, _ = np.linalg.lstsq(design_matrix, world_pts_np[:, 0], rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(design_matrix, world_pts_np[:, 1], rcond=None)

    return (*params_x, *params_y), utm_crs


def apply_affine_transform(x: float, y: float, z: float, affine_params, ref_ele: float, utm_crs):
    """
    Transform Metashape local coordinates to WGS84.

    The affine fit is applied in the X/Z plane; elevation is derived by anchoring
    to the first reference's world elevation.
    """
    a, b, c, d, e, f = affine_params
    utm_e = a * x + b * z + c
    utm_n = d * x + e * z + f

    wgs84_crs = CRS.from_epsg(4326)
    inverse_transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
    lon, lat = inverse_transformer.transform(utm_e, utm_n)
    elevation = ref_ele - y
    return lon, lat, elevation


# --------------------------------------------------------------------------- #
# Lambda entry point
# --------------------------------------------------------------------------- #
def lambda_handler(event, _context):
    """
    Entry point: read references and trajectory, fit affine, emit geo-posed file.

    Expected event payload:
        {
            "bucket": "...",
            "input_key": "...",
            "output_key": "...",
            "manual_refs": {
                "0": [lat, lon, elev],
                "15": [...],
                ...
            }
        }
    """
    try:
        bucket = event.get('bucket')
        input_key = event.get('input_key')
        output_key = event.get('output_key')
        manual_refs = event.get('manual_refs') or {}

        if not all([bucket, input_key, output_key, manual_refs]):
            return {'statusCode': 400, 'body': json.dumps('Missing required parameters')}

        trajectories = read_trajectory_from_s3(bucket, input_key)
        if not trajectories:
            return {'statusCode': 400, 'body': json.dumps('No valid trajectory data found')}

        affine_params, utm_crs = calculate_affine_params(manual_refs, trajectories)
        axes_matrix = build_local_to_world_axes(affine_params)

        first_ref_idx = min(manual_refs.keys(), key=lambda key: int(key))
        _, _, first_ele = manual_refs[first_ref_idx]

        output = StringIO()
        output.write("photo_id longitude latitude elevation roll pitch yaw\n")

        for trajectory in trajectories:
            try:
                lon, lat, ele = apply_affine_transform(
                    trajectory['x'],
                    trajectory['y'],
                    trajectory['z'],
                    affine_params,
                    first_ele,
                    utm_crs,
                )

                camera_axes_world = axes_matrix @ trajectory['rotation_matrix']
                for axis in range(3):
                    axis_world = camera_axes_world[:, axis]
                    axis_norm = np.linalg.norm(axis_world)
                    if axis_norm > 1e-8:
                        camera_axes_world[:, axis] = axis_world / axis_norm

                forward_world = camera_axes_world[:, 2]
                east = forward_world[0]
                north = forward_world[1]
                horizontal_norm = np.hypot(east, north)
                if horizontal_norm < 1e-8:
                    yaw = 0.0
                else:
                    yaw = wrap_to_180(np.degrees(np.arctan2(east, north)))

                roll = wrap_to_180(-trajectory['omega'])
                pitch = wrap_to_180(-trajectory['kappa'])

                if np.isfinite(lon) and np.isfinite(lat):
                    output.write(
                        f"{trajectory['photo_id']} {lon:.8f} {lat:.8f} {ele:.3f} "
                        f"{roll:.3f} {pitch:.3f} {yaw:.3f}\n"
                    )
                else:
                    print(f"Warning: invalid coordinates for {trajectory['photo_id']}")
            except Exception as exc:
                print(f"Error processing trajectory {trajectory['photo_id']}: {exc}")

        s3.put_object(Bucket=bucket, Key=output_key, Body=output.getvalue())
        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Conversion completed successfully using affine transformation'}),
        }

    except Exception as exc:
        print(f"Error in lambda function: {exc}")
        return {'statusCode': 500, 'body': json.dumps(f'Error: {str(exc)}')}