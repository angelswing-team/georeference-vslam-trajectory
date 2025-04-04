import json
import boto3
import numpy as np
from pyproj import CRS, Transformer
from io import StringIO
import math

# Initialize S3 client
s3 = boto3.client('s3')

def read_trajectory_from_s3(bucket, key):
    """Read SLAM trajectory file from S3"""
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')

        trajectories = []
        for line in content.splitlines():
            parts = line.strip().split()
            if len(parts) == 8:
                timestamp = float(parts[0])
                x = float(parts[1])  # Forward
                y = float(parts[2])  # Left
                z = float(parts[3])  # Up
                qx = float(parts[4])
                qy = float(parts[5])
                qz = float(parts[6])
                qw = float(parts[7])
                trajectories.append((timestamp, x, y, z, qx, qy, qz, qw))

        return trajectories
    except Exception as e:
        print(f"Error reading trajectory file from S3: {e}")
        raise

def calculate_transform_params(first_lat, first_lon, last_lat, last_lon, trajectories):
    """Calculate rotation angle and scale between SLAM and global coordinates"""
    # Create UTM projection for distance calculation
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

    # Calculate GPS track length
    first_e, first_n = transformer.transform(first_lon, first_lat)
    last_e, last_n = transformer.transform(last_lon, last_lat)
    gps_dx = last_e - first_e
    gps_dy = last_n - first_n
    gps_length = np.sqrt(gps_dx**2 + gps_dy**2)

    # Calculate SLAM track length
    last_point = trajectories[-1]
    slam_dx = last_point[1]  # x coordinate
    slam_dy = last_point[3]  # z coordinate (forward direction)
    slam_length = np.sqrt(slam_dx**2 + slam_dy**2)

    # Calculate scale factor
    scale = gps_length / slam_length if slam_length > 0 else 1.0

    # Calculate rotation angle
    gpx_vector = np.array([gps_dx, gps_dy])
    gpx_vector = gpx_vector / np.linalg.norm(gpx_vector)

    slam_vector = np.array([slam_dx, slam_dy])
    slam_vector = slam_vector / np.linalg.norm(slam_vector)

    angle = np.arctan2(np.cross(slam_vector, gpx_vector), np.dot(slam_vector, gpx_vector))

    print(f"GPS track length: {gps_length:.2f}m")
    print(f"SLAM track length: {slam_length:.2f}m")
    print(f"Scale factor: {scale:.4f}")
    print(f"Rotation angle: {np.degrees(angle):.2f} degrees")

    return angle, scale

def convert_coordinates(x, y, z, ref_lat, ref_lon, ref_ele, rotation_angle=0, scale=1.0):
    """Convert local coordinates from Stella-SLAM to global coordinates with rotation and scale"""
    # Define CRS explicitly
    utm_zone = int((ref_lon + 180) / 6) + 1
    hemisphere = 'north' if ref_lat >= 0 else 'south'

    # Create CRS objects
    wgs84_crs = CRS.from_epsg(4326)
    utm_crs = CRS.from_dict({
        'proj': 'utm',
        'zone': utm_zone,
        'hemisphere': hemisphere,
        'ellps': 'WGS84',
        'datum': 'WGS84',
        'units': 'm'
    })

    # Create transformer
    transformer = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)
    inverse_transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)

    try:
        # Convert reference point to UTM
        ref_e, ref_n = transformer.transform(ref_lon, ref_lat)

        # Apply scale and rotation to x,z coordinates
        x_scaled = x * scale
        z_scaled = z * scale

        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)
        x_rotated = x_scaled * cos_angle - z_scaled * sin_angle
        z_rotated = x_scaled * sin_angle + z_scaled * cos_angle

        # Convert coordinates
        utm_e = ref_e + x_rotated
        utm_n = ref_n + z_rotated
        elevation = ref_ele - y

        # Convert back to WGS84
        lon, lat = inverse_transformer.transform(utm_e, utm_n)
        return lon, lat, elevation

    except Exception as e:
        print(f"Error in coordinate conversion: {e}")
        return ref_lon, ref_lat, ref_ele

def quaternion_to_euler(qx, qy, qz, qw):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw) following aerospace convention.
    
    Standard aerospace convention:
    - Roll: Rotation around X axis (front-to-back)
    - Pitch: Rotation around Y axis (side-to-side) 
    - Yaw: Rotation around Z axis (vertical)
    
    Args:
        qx, qy, qz, qw: Quaternion components
        
    Returns:
        roll, pitch, yaw angles in degrees
    """
    # Check if quaternion is normalized
    norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    if abs(norm - 1.0) > 1e-3:
        # Normalize the quaternion
        qw /= norm
        qx /= norm
        qy /= norm
        qz /= norm
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        # Use 90 degrees if out of range
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # Convert to degrees
    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)
    
    return roll_deg, pitch_deg, yaw_deg

def lambda_handler(event, context):
    """
    AWS Lambda handler function

    Expected event format:
    {
        "bucket": "dev-storage.angelswing.io",
        "input_key": "videos/1398/5000/trajectory/keyframe_trajectory.txt",
        "output_key": "videos/1398/5000/trajectory/geo_trajectory.txt",
        "manual_refs": {
            "first": [37.237346666666674,127.2938166666666,170.9],
            "last": [37.2366433396141,127.29341833301352,157.301]
        }
    }
    """
    try:
        # Parse input parameters
        bucket = event.get('bucket')
        input_key = event.get('input_key')
        output_key = event.get('output_key')
        manual_refs = event.get('manual_refs')

        if not all([bucket, input_key, output_key, manual_refs]):
            return {
                'statusCode': 400,
                'body': json.dumps('Missing required parameters')
            }

        # Extract reference coordinates
        first_ref = manual_refs.get('first')
        last_ref = manual_refs.get('last')

        if not first_ref or not last_ref or len(first_ref) != 3 or len(last_ref) != 3:
            return {
                'statusCode': 400,
                'body': json.dumps('Invalid reference coordinates format')
            }

        first_lat, first_lon, first_ele = first_ref
        last_lat, last_lon, last_ele = last_ref

        # Log configuration
        print(f"Processing file {input_key} to {output_key}")
        print(f"First reference: lat={first_lat}, lon={first_lon}, ele={first_ele}")
        print(f"Last reference: lat={last_lat}, lon={last_lon}, ele={last_ele}")
        print(f"UTM Zone: {int((first_lon + 180) / 6) + 1}")

        # Read trajectory data
        trajectories = read_trajectory_from_s3(bucket, input_key)

        if not trajectories:
            return {
                'statusCode': 400,
                'body': json.dumps('No valid trajectory data found')
            }

        # Calculate transformation parameters
        rotation_angle, scale = calculate_transform_params(
            first_lat, first_lon, last_lat, last_lon, trajectories
        )

        # Convert coordinates and prepare output
        output = StringIO()
        output.write("timestamp longitude latitude elevation roll pitch yaw\n")

        for timestamp, x, y, z, qx, qy, qz, qw in trajectories:
            try:
                lon, lat, ele = convert_coordinates(
                    x, y, z, first_lat, first_lon, first_ele, rotation_angle, scale
                )
                
                # Calculate roll, pitch, yaw angles from quaternion
                roll, pitch, yaw = quaternion_to_euler(qx, qy, qz, qw)

                if np.isfinite(lon) and np.isfinite(lat):
                    output.write(f"{timestamp:.3f} {lon} {lat} {ele:.2f} {roll:.2f} {pitch:.2f} {yaw:.2f}\n")
                else:
                    print(f"Warning: Invalid coordinates generated for point: x={x}, y={y}, z={z}")
            except Exception as e:
                print(f"Error processing point: {e}")

        # Upload result to S3
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=output.getvalue()
        )

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Conversion completed successfully',
            })
        }

    except Exception as e:
        print(f"Error in lambda function: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }