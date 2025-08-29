import numpy as np
import random
import cv2

from mask_simulation import get_random_shapes
from mask_simulation import get_object_shapes
from mask_simulation import get_translation_rotation_matrix, get_random_affine_matrix, apply_affine_transform


def random_bezier_path(w, h, f, num_control_points=4, max_length=None, init_position=None):
    control_points = np.random.rand(num_control_points, 2) * np.array([w, h])
   
    if init_position is not None:
        control_points[0] = np.array(init_position)

    def de_casteljau(points, t):
        while len(points) > 1:
            points = [(1 - t) * p0 + t * p1 for p0, p1 in zip(points[:-1], points[1:])]
        return points[0]

    # Step 1: Generate high-res path for accurate arc-length sampling
    high_res = 1000
    fine_ts = np.linspace(0, 1, high_res)
    fine_path = np.array([de_casteljau(control_points, t) for t in fine_ts])

    # Step 2: Compute cumulative arc length
    segment_lengths = np.linalg.norm(np.diff(fine_path, axis=0), axis=1)
    cum_lengths = np.insert(np.cumsum(segment_lengths), 0, 0)

    # Step 3: Limit total length
    total_length = cum_lengths[-1]
    if max_length is not None:
        total_length = min(total_length, max_length)

    # Step 4: Generate f evenly spaced lengths within the max length
    target_lengths = np.linspace(0, total_length, f + 1)

    # Step 5: Interpolate the fine_path at those arc lengths
    sampled_path = []
    for tl in target_lengths:
        idx = np.searchsorted(cum_lengths, tl)
        if idx == 0:
            sampled_path.append(fine_path[0])
        elif idx >= len(cum_lengths):
            sampled_path.append(fine_path[-1])
        else:
            # Linear interpolation between points
            t = (tl - cum_lengths[idx - 1]) / (cum_lengths[idx] - cum_lengths[idx - 1])
            p1, p2 = fine_path[idx - 1], fine_path[idx]
            interp = (1 - t) * p1 + t * p2
            sampled_path.append(interp)
    sampled_path = np.array(sampled_path)
    return sampled_path[1:] - sampled_path[:-1]


def simu_motion_deform_bezier(mask, width=480, height=640, max_length=480, v_angle_max=5, num_frames=14, init_position=None):
    """Simulate the motion and deformation of the mask"""
    mask_sequence = []
    angle_max = v_angle_max / 180 * np.pi
    angle = np.random.uniform(-angle_max, angle_max)

    path = random_bezier_path(width, height, num_frames, num_control_points=np.random.randint(4, 8), max_length=max_length, init_position=init_position)
    for i in range(num_frames):
        angle = angle * np.random.choice([1, -1], p=[0.8, 0.2])
        matrix_distort = get_random_affine_matrix(
            width=width,
            height=height,
            tx=path[i][0],
            ty=path[i][1],
            angle=angle,
            scale_min=0.995,
            scale_max=1.005,
            aspect_ratio_min=0.95,
            aspect_ratio_max=1.05,
            shear_max=0.01,
        )
        mask = apply_affine_transform(mask, matrix=matrix_distort)
        mask_sequence.append(mask)
    return mask_sequence

def simu_motion_deform(mask, width=480, height=640, vx_max=10, vy_max=10, v_angle_max=5, num_frames=14):
    """Simulate the motion and deformation of the mask"""
    mask_sequence = []
    vx = np.random.uniform(-vx_max, vx_max)
    vy = np.random.uniform(-vy_max, vy_max)
    v_angle = np.random.uniform(-v_angle_max, v_angle_max)
    matrix_motion = get_translation_rotation_matrix(width=width, height=height, vx=vx, vy=vy, v_angle=v_angle)
    for i in range(num_frames):
        matrix_distort = get_random_affine_matrix(width=width, height=height, tx_max=0.01, ty_max=0.01, angle_max=np.pi * 0.01, scale_min=0.995, scale_max=1.005, aspect_ratio_min=0.95, aspect_ratio_max=1.05, shear_max=0.01)
        mask = apply_affine_transform(mask, matrix=matrix_motion @ matrix_distort)
        mask_sequence.append(mask)
    return mask_sequence


def generate_multi_mask_instances(num_instances_max=3, num_frames=24, height=480, width=360, smoothen=True, server="local"):
    mask_sequence_out = [np.zeros((height, width), dtype=np.uint8) for _ in range(num_frames)]
    num_instances = np.random.randint(20, max(21, num_instances_max))

    for i in range(num_instances):
        random_choice = np.random.choice([1, 2], p=[0.5, 0.5])
        if random_choice == 1 or i >= 2:
            if i < 2:
                #ratios = [np.random.uniform(0.1, 1), np.random.uniform(0.1, 1)]
                ratios = [1, 1]
            else:
                ratios = [np.random.uniform(0.02, 0.2), np.random.uniform(0.02, 0.2)]
            obj = get_random_shapes(width=int(width * ratios[0]), height=int(height * ratios[1]))

            ph = 0 if height == obj.shape[0] else np.random.randint(0, height - obj.shape[0])
            pw = 0 if width == obj.shape[1] else np.random.randint(0, width - obj.shape[1])
        elif random_choice == 2:
            obj = get_object_shapes(width=width, height=height, server=server, random_affine=True)
            ph, pw = 0, 0
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[ph : ph + obj.shape[0], pw : pw + obj.shape[1]] = obj
        # else:
        #     mask = get_coco_shapes(width=width, height=height)
        #mask_sequence_tmp = simu_motion_deform(mask, width=width, height=height, num_frames=num_frames, max_length=max_length, init_position=[ph + obj.shape[0] // 2, pw + obj.shape[1] // 2])
        mask_sequence_tmp = simu_motion_deform(mask, width=width, height=height, num_frames=num_frames)
        # Add the new mask instances
        for j in range(num_frames):
            mask_sequence_out[j] = np.maximum(mask_sequence_out[j], mask_sequence_tmp[j])
            if smoothen:  # smmothen the mask boundaries
                # mask_sequence_out[j] = cv2.GaussianBlur(mask_sequence_out[j], (9, 9), 0)
                kernel_size = 5
                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                mask_sequence_out[j] = cv2.morphologyEx(mask_sequence_out[j], cv2.MORPH_CLOSE, kernel)
    return mask_sequence_out


if __name__ == "__main__":
    from moviepy.editor import ImageSequenceClip

    for i in range(100):
        # Generate multiple mask instances
        mask_sequence = generate_multi_mask_instances(num_instances_max=30, num_frames=49, height=512, width=512)

        # Export the video
        # Create a video clip from the numpy array of images
        mask3ch_sequence = [cv2.merge([mask, mask, mask]) for mask in mask_sequence]
        clip = ImageSequenceClip(mask3ch_sequence, fps=15)  # Adjust fps (frames per second) as needed
        path_mask_sequence = f"samples/mask_sequence_{i}.mp4"
        clip.write_videofile(path_mask_sequence, codec="libx264")
