import os
import boto3
import numpy as np
from pyproj import CRS, Transformer
from io import StringIO
import functions_framework
import math

# Initialize GCS client using boto3 S3-compatible API
s3 = boto3.client(
    "s3",
    endpoint_url="https://storage.googleapis.com",
    region_name="auto",
    aws_access_key_id=os.getenv("GOOGLE_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("GOOGLE_SECRET_ACCESS_KEY"),
)

def read_trajectory_from_s3(bucket, key):
    """Read SLAM trajectory file from Google Cloud Storage"""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')

        trajectories = []
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) == 8:
                trajectories.append(tuple(map(float, parts)))

        return trajectories
    except Exception as e:
        print(f"Error reading trajectory file from GCS: {e}")
        raise

def calculate_affine_params(manual_refs, trajectories):
    """
    Calculate the 6 parameters of a 2D affine transformation using a least-squares fit
    on multiple reference points mapped from local SLAM (x,z) to world UTM (easting,northing).

    Returns (a,b,c,d,e,f), utm_crs such that:
      UTM_E = a*X + b*Z + c
      UTM_N = d*X + e*Z + f
    """
    if not isinstance(manual_refs, dict) or len(manual_refs) < 3:
        raise ValueError("At least 3 reference points are required for an affine transformation.")

    # Determine UTM from the first reference (smallest index)
    try:
        first_ref_idx = sorted(manual_refs.keys(), key=lambda k: int(k))[0]
    except Exception:
        first_ref_idx = list(manual_refs.keys())[0]

    first_lat, first_lon, _ = manual_refs[first_ref_idx]

    utm_zone = int((first_lon + 180) / 6) + 1
    hemisphere = 'north' if first_lat >= 0 else 'south'
    wgs84_crs = CRS.from_epsg(4326)
    utm_crs = CRS.from_dict({
        'proj': 'utm',
        'zone': utm_zone,
        'hemisphere': hemisphere,
        'ellps': 'WGS84',
        'datum': 'WGS84',
        'units': 'm'
    })
    transformer = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)

    local_points = []
    world_points_utm = []

    for idx_key, (lat, lon, _ele) in manual_refs.items():
        try:
            idx = int(idx_key)
        except Exception:
            raise ValueError("Manual reference keys must be numeric indices as strings, e.g., '0','5','42'.")

        # trajectories[idx] = (timestamp, x, y, z, qx, qy, qz, qw)
        _, slam_x, _, slam_z, _, _, _, _ = trajectories[idx]
        local_points.append([slam_x, slam_z])

        utm_e, utm_n = transformer.transform(lon, lat)
        world_points_utm.append([utm_e, utm_n])

    local_pts_np = np.array(local_points)
    world_pts_np = np.array(world_points_utm)

    A = np.hstack([local_pts_np, np.ones((local_pts_np.shape[0], 1))])

    params_x, _, _, _ = np.linalg.lstsq(A, world_pts_np[:, 0], rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(A, world_pts_np[:, 1], rcond=None)

    a, b, c = params_x
    d, e, f = params_y

    print("Calculated Affine Transformation Parameters:")
    print(f"a={a:.4f}, b={b:.4f}, c={c:.4f}")
    print(f"d={d:.4f}, e={e:.4f}, f={f:.4f}")

    return (float(a), float(b), float(c), float(d), float(e), float(f)), utm_crs

def apply_affine_transform(x, y, z, affine_params, ref_ele, utm_crs):
    """Convert local SLAM coordinates to global coordinates using affine parameters."""
    a, b, c, d, e, f = affine_params

    wgs84_crs = CRS.from_epsg(4326)
    inverse_transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

    utm_e = a * x + b * z + c
    utm_n = d * x + e * z + f

    elevation = ref_ele - y

    lon, lat = inverse_transformer.transform(utm_e, utm_n)
    return lon, lat, elevation

def quaternion_to_euler(qx, qy, qz, qw):
    """Convert quaternion to Euler angles (roll, pitch, yaw) in degrees."""
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

@functions_framework.http
def convert_trajectory(request):
    """
    Cloud Run HTTP endpoint handler using Functions Framework

    Expected request JSON format:
    {
        "bucket": "dev-storage.angelswing.io",
        "input_key": "videos/1398/5000/trajectory/keyframe_trajectory.txt",
        "output_key": "videos/1398/5000/trajectory/geo_trajectory.txt",
        "manual_refs": {
            "0": [lat, lon, ele],
            "12": [lat, lon, ele],
            "42": [lat, lon, ele]
        }
    }
    """
    try:
        event = request.get_json()

        # Parse input parameters
        bucket = event.get('bucket')
        input_key = event.get('input_key')
        output_key = event.get('output_key')
        manual_refs = event.get('manual_refs')

        if not all([bucket, input_key, output_key, manual_refs]):
            return ({'error': 'Missing required parameters'}, 400)

        # Validate reference points (require at least 3 for affine transformation)
        if not isinstance(manual_refs, dict) or len(manual_refs) < 3:
            return ({'error': 'At least 3 reference points are required in manual_refs'}, 400)

        # Log configuration
        print(f"Processing file {input_key} to {output_key}")
        print(f"Manual reference points provided: {len(manual_refs)}")

        # Read trajectory data
        trajectories = read_trajectory_from_s3(bucket, input_key)

        if not trajectories:
            return ({'error': 'No valid trajectory data found'}, 400)

        # Calculate affine transformation parameters from all reference points
        affine_params, utm_crs = calculate_affine_params(manual_refs, trajectories)

        # Derive effective rotation from affine (used for yaw correction)
        a, _b, _c, d, _e, _f = affine_params
        effective_rotation_angle = np.arctan2(d, a)

        # Convert coordinates and prepare output
        output = StringIO()
        output.write("timestamp longitude latitude elevation roll pitch yaw\n")

        # Use the first reference's elevation as baseline
        try:
            first_ref_idx = sorted(manual_refs.keys(), key=lambda k: int(k))[0]
        except Exception:
            first_ref_idx = list(manual_refs.keys())[0]
        _lat0, _lon0, first_ele = manual_refs[first_ref_idx]

        for timestamp, x, y, z, qx, qy, qz, qw in trajectories:
            try:
                # Affine transform of (x,z) to lon/lat, with elevation from y
                lon, lat, ele = apply_affine_transform(
                    x, y, z, affine_params, first_ele, utm_crs
                )

                # Transform quaternion from Y-up (SLAM) to Z-up (geospatial)
                temp_qy = qy
                qy_new = -qz
                qz_new = temp_qy

                # Convert quaternion to Euler and correct yaw by effective rotation
                roll, pitch, yaw = quaternion_to_euler(qx, qy_new, qz_new, qw)
                yaw = yaw - np.degrees(effective_rotation_angle)

                if np.isfinite(lon) and np.isfinite(lat):
                    output.write(f"{timestamp:.3f} {lon:.8f} {lat:.8f} {ele:.3f} {roll:.3f} {pitch:.3f} {yaw:.3f}\n")
                else:
                    print(f"Warning: Invalid coordinates generated for point: x={x}, y={y}, z={z}")
            except Exception as e:
                print(f"Error processing point: {e}")

        # Upload result to Google Cloud Storage
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=output.getvalue()
        )

        return ({'message': 'Conversion completed successfully using affine transformation'}, 200)

    except Exception as e:
        print(f"Error in function: {e}")
        return ({'error': str(e)}, 500)