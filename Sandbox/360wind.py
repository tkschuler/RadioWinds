import matplotlib.pyplot as plt
import numpy as np

def count_unique_servo_circles(angles):
    unique_circles = set()
    clockwise_circles = 0
    counterclockwise_circles = 0
    prev_angle = None
    prev_direction = None

    for angle in angles:
        if prev_angle is not None:
            # Determine the angle difference between the current and previous angles
            angle_diff = (angle - prev_angle + 180) % 360 - 180  # Correct for crossing 0/360-degree axis

            if angle_diff > 0:
                direction = "clockwise"
            else:
                direction = "counterclockwise"

            # Check if a complete circle is formed
            if prev_direction is not None and direction != prev_direction:
                circle = (min(prev_angle, angle), max(prev_angle, angle))
                unique_circles.add(circle)
                if direction == "clockwise":
                    clockwise_circles += 1
                else:
                    counterclockwise_circles += 1

            prev_direction = direction

        prev_angle = angle

    return len(unique_circles), clockwise_circles, counterclockwise_circles


def plot_interpolated_polar_angles(angles, num_interpolated_points=100):
    # Prepare data for plotting
    angles = np.array(angles)
    angles_radians = np.deg2rad(angles)

    interpolated_angles = []
    interpolated_radii = []

    for i in range(len(angles) - 1):
        start_angle = angles_radians[i] % 2*np.pi
        end_angle = angles_radians[i + 1] % 2*np.pi
        angle_diff = abs(end_angle - start_angle)

        # Correct for crossing the 0/360-degree axis
        if angle_diff > np.pi:
            if (end_angle > start_angle):
                start_angle += 2*np.pi
            else:
                end_angle += 2 * np.pi

        # Linearly interpolate 100 points between the angles
        for j in range(num_interpolated_points + 1):
            t = j / num_interpolated_points
            #interp_angle = start_angle + t * angle_diff
            interp_angle = np.interp(j+t, [j, j+1], [start_angle, end_angle]) % 2*np.pi
            interp_radius = i + t
            interpolated_angles.append(interp_angle)
            interpolated_radii.append(interp_radius)

    # Create a polar scatter plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(interpolated_angles, interpolated_radii, s=5, marker='o', color='b')

    # Set theta labels
    ax.set_xticks(np.deg2rad([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['0', '45', '90', '135', '180', '225', '270', '315'])

    plt.title("Polar Scatter Plot with Interpolated Values")

def plot_servo_angles(angles, circle_info, clockwise_color='g', counterclockwise_color='r'):
    # Extract circle information
    circle_indices, clockwise_indices, counterclockwise_indices = circle_info
    circle_angles = [angles[i] for i in circle_indices]
    clockwise_angles = [angles[i] for i in clockwise_indices]
    counterclockwise_angles = [angles[i] for i in counterclockwise_indices]

    # Create a scatter plot of all angles
    plt.figure(figsize=(8, 6))
    plt.scatter(np.arange(len(angles)), angles, c='b', label='Non-Circle Angles')

    # Highlight clockwise and counterclockwise circles
    plt.scatter(circle_indices, circle_angles, c='k', marker='o', label='Complete Circles')
    plt.scatter(clockwise_indices, clockwise_angles, c=clockwise_color, marker='o', label='Clockwise Circles')
    plt.scatter(counterclockwise_indices, counterclockwise_angles, c=counterclockwise_color, marker='o', label='Counterclockwise Circles')

    # Add labels and legend
    plt.xlabel('Sample Index')
    plt.ylabel('Angle (degrees)')
    plt.legend()

    # Show the plot
    plt.title('Servo Angle Visualization')
    plt.show()

# Example usage:
import random

# Simulate servo movements to 20 different angles
def generate_random_angles(num_angles, max_diff=10):
    angles = [random.randint(0, 359)]
    for _ in range(num_angles - 1):
        min_angle = max(angles[-1] - max_diff, 0)
        max_angle = min(angles[-1] + max_diff, 359)
        new_angle = random.randint(min_angle, max_angle)
        angles.append(new_angle)
    return angles

# Example usage:
angles = generate_random_angles(10, max_diff=10)


plot_interpolated_polar_angles(angles, num_interpolated_points=1000)
print (angles)

unique_circles, clockwise_circles, counterclockwise_circles = count_unique_servo_circles(angles)
print(f"Number of unique complete circles: {unique_circles}")
print(f"Number of clockwise circles: {clockwise_circles}")
print(f"Number of counterclockwise circles: {counterclockwise_circles}")


plt.show()
