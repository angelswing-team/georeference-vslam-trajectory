import json
import boto3
import numpy as np
from pyproj import CRS, Transformer
from io import StringIO

# Initialize S3 client
s3 = boto3.client('s3')

def read_trajectory_from_s3(bucket, key):
    """Read trajectory file with Euler angles from S3"""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')

        trajectories = []
        for line in content.splitlines():
            line = line.strip()
            if line.startswith('#') or not line:
                continue  # Skip comments and empty lines

            parts = line.split('\t')  # Tab-separated values
            if len(parts) >= 10:  # PhotoID, X, Y, Z, Omega, Phi, Kappa, and rotation matrix
                try:
                    photo_id = parts[0]
                    x, y, z = map(float, parts[1:4])
                    omega, phi, kappa = map(float, parts[4:7])
                    # Return: photo_id, x, y, z, omega, phi, kappa
                    trajectories.append((photo_id, x, y, z, omega, phi, kappa))

                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping invalid line: {line} - {e}")
                    continue

        return trajectories
    except Exception as e:
        print(f"Error reading trajectory file from S3: {e}")
        raise

def calculate_affine_params(manual_refs, trajectories):
    """
    Calculate the 6 parameters of an affine transformation using a least-squares fit
    on multiple reference points.
    """
    # Validate that we have enough points to solve
    if len(manual_refs) < 3:
        raise ValueError("At least 3 reference points are required for an affine transformation.")

    # Get the first reference lon to determine the UTM zone for consistency
    first_ref_idx = sorted(manual_refs.keys(), key=int)[0]
    first_lat, first_lon, _ = manual_refs[first_ref_idx]

    # Create UTM projection for all points
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

    # Prepare control points
    for idx_str, (lat, lon, ele) in manual_refs.items():
        idx = int(idx_str)
        # SLAM local coordinates (x, z are used for 2D plane)
        # Note: trajectories list is 0-indexed, but reference indices are 1-indexed frame numbers
        trajectory_idx = idx - 1 if idx > 0 else 0
        _, slam_x, _, slam_z, _, _, _ = trajectories[trajectory_idx]
        local_points.append([slam_x, slam_z])

        # World coordinates converted to UTM
        utm_e, utm_n = transformer.transform(lon, lat)
        world_points_utm.append([utm_e, utm_n])

    local_pts_np = np.array(local_points)
    world_pts_utm_np = np.array(world_points_utm)

    # Pad local points with a column of ones for the affine transformation matrix
    A = np.hstack([local_pts_np, np.ones((local_pts_np.shape[0], 1))])

    # Solve for the transformation parameters (a, d, c) and (b, e, f) using least squares
    # This finds the best fit if more than 3 points are provided.
    # World_X = a*Local_X + b*Local_Y + c
    # World_Y = d*Local_X + e*Local_Y + f
    # Note: Our local_y is slam_z
    params_x, _, _, _ = np.linalg.lstsq(A, world_pts_utm_np[:, 0], rcond=None)
    params_y, _, _, _ = np.linalg.lstsq(A, world_pts_utm_np[:, 1], rcond=None)

    # Parameters: a, b, c, d, e, f
    a, b, c = params_x
    d, e, f = params_y

    print(f"Calculated Affine Transformation Parameters:")
    print(f"a={a:.4f}, b={b:.4f}, c={c:.4f}")
    print(f"d={d:.4f}, e={e:.4f}, f={f:.4f}")

    return (a, b, c, d, e, f), utm_crs

def apply_affine_transform(x, y, z, affine_params, ref_ele, utm_crs):
    """Convert local SLAM coordinates to global coordinates using affine parameters."""
    # Unpack parameters
    a, b, c, d, e, f = affine_params

    # Create the inverse transformer to convert from UTM back to WGS84
    wgs84_crs = CRS.from_epsg(4326)
    inverse_transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

    # Apply the 2D affine transformation to the x, z plane
    # Note: SLAM's forward direction 'z' corresponds to the second dimension in our 2D plane
    utm_e = a * x + b * z + c
    utm_n = d * x + e * z + f

    # Handle elevation simply based on reference and SLAM's y-axis
    elevation = ref_ele - y

    # Convert the transformed UTM coordinates back to WGS84 (lon, lat)
    lon, lat = inverse_transformer.transform(utm_e, utm_n)

    return lon, lat, elevation

def lambda_handler(event, context):
    """
    AWS Lambda handler function for georeferencing SLAM trajectories
    using a multi-point affine transformation.

        Expected event format:
    {
        "bucket": "dev-storage.angelswing.io",
        "input_key": "videos/1398/5000/trajectory/keyframe_trajectory.txt",
        "output_key": "videos/1398/5000/trajectory/geo_trajectory.txt",
        "manual_refs": {
            "0": [37.237346666666674,127.2938166666666,170.9],
            "index": [37.2366433396141,127.29341833301352,157.301],
            "last_index": [37.2366433396141,127.29341833301352,157.301]
        }
    }
    """
    try:
        bucket = event.get('bucket')
        input_key = event.get('input_key')
        output_key = event.get('output_key')
        manual_refs = event.get('manual_refs')

        if not all([bucket, input_key, output_key, manual_refs]):
            return {'statusCode': 400, 'body': json.dumps('Missing required parameters')}

        # Log configuration
        print(f"Processing file {input_key} to {output_key}")
        print(f"Manual reference points provided: {len(manual_refs)}")

        # Read trajectory data from S3
        trajectories = read_trajectory_from_s3(bucket, input_key)
        if not trajectories:
            return {'statusCode': 400, 'body': json.dumps('No valid trajectory data found')}

        # Calculate transformation parameters using all reference points
        affine_params, utm_crs = calculate_affine_params(manual_refs, trajectories)

        # Calculate the effective rotation of the transformation for yaw correction.
        # This angle represents how the new X-axis is oriented.
        a, _, _, d, _, _ = affine_params
        effective_rotation_angle = np.arctan2(d, a)
        print(f"Effective rotation angle for yaw correction: {np.degrees(effective_rotation_angle):.2f} degrees")

        # Get the first reference point for initial elevation
        first_ref_idx = sorted(manual_refs.keys(), key=int)[0]
        _, _, first_ele = manual_refs[first_ref_idx]

        # Convert coordinates and prepare output file
        output = StringIO()
        output.write("photo_id longitude latitude elevation roll pitch yaw\n")

        for photo_id, x, y, z, omega, phi, kappa in trajectories:
            try:
                lon, lat, ele = apply_affine_transform(
                    x, y, z, affine_params, first_ele, utm_crs
                )
                # roll, pitch, yaw = omega, phi, kappa
                roll, pitch, yaw = -omega, -kappa, -phi

                yaw = yaw - np.degrees(effective_rotation_angle)

                if np.isfinite(lon) and np.isfinite(lat):
                    output.write(f"{photo_id:.3f} {lon:.8f} {lat:.8f} {ele:.3f} {roll:.3f} {pitch:.3f} {yaw:.3f}\n")
                else:
                    print(f"Warning: Invalid coordinates generated for point: x={x}, y={y}, z={z}")
            except Exception as e:
                print(f"Error processing point ({x},{y},{z}): {e}")

        # Upload the georeferenced trajectory to S3
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=output.getvalue()
        )

        return {
            'statusCode': 200,
            'body': json.dumps({'message': 'Conversion completed successfully using affine transformation'})
        }

    except Exception as e:
        print(f"Error in lambda function: {e}")
        return {'statusCode': 500, 'body': json.dumps(f'Error: {str(e)}')}