import numpy as np
from skimage.transform import warp
from skimage.transform import AffineTransform, ProjectiveTransform


def get_translation_rotation_matrix(width=512, height=512, vx=2, vy=2, v_angle=1):
    """vx, vy: translation in pixels
    v_angle: rotation in degrees"""
    # Create a transformation matrix for translation and rotation
    matrix_motion = AffineTransform(translation=(vx, vy), rotation=np.deg2rad(v_angle))
    # Translation matrices
    center_y, center_x = height // 2, width // 2
    T_translate_to_center = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])
    T_translate_back = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])
    matrix_motion = T_translate_back @ matrix_motion @ T_translate_to_center
    return matrix_motion


def get_random_affine_matrix(width=512, height=512, tx_max=0.25, ty_max=0.25, angle_max=np.pi * 0.25, scale_min=0.1, scale_max=0.5, aspect_ratio_min=0.5, aspect_ratio_max=2.0, shear_max=0.1):
    # Random translation
    tx = np.random.uniform(-width * tx_max, width * tx_max)
    ty = np.random.uniform(-height * ty_max, height * ty_max)
    # Random rotation
    angle = np.random.uniform(-angle_max, angle_max)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    # Random scaling
    scale = np.random.uniform(scale_min, scale_max)
    aspect_ratio = np.random.uniform(aspect_ratio_min, aspect_ratio_max)
    sx = scale * 1.0
    sy = scale * aspect_ratio
    # Random shearing
    shear = np.random.uniform(-shear_max, shear_max)
    # Translation matrices
    center_y, center_x = height // 2, width // 2
    T_translate_to_center = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])
    T_translate_back = np.array([[1, 0, center_x], [0, 1, center_y], [0, 0, 1]])
    # Affine transformation matrix
    affine_matrix = np.array(
        [
            [sx * cos_angle, -sy * sin_angle + shear, tx],
            [sx * sin_angle, sy * cos_angle + shear, ty],
            [0, 0, 1],
        ]
    )
    combined_matrix = T_translate_back @ affine_matrix @ T_translate_to_center
    return combined_matrix


def apply_random_perspective(image, max_warp=0.2):
    height, width = image.shape[:2]
    # Define original corners of the image
    src = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    # Generate random displacement for each corner
    displacement = np.random.uniform(-max_warp, max_warp, size=(4, 2)) * [width, height]
    dst = src + displacement
    # Create a ProjectiveTransform object
    transform = ProjectiveTransform()
    transform.estimate(src, dst)
    # Apply the perspective transform
    warped_image = warp(image, transform, output_shape=(height, width))
    warped_image = (warped_image * 255).astype(np.uint8)
    return warped_image


def apply_affine_transform(image, matrix):
    transform = AffineTransform(matrix=matrix)
    warped_image = warp(image, transform.inverse, output_shape=image.shape)
    warped_image = (warped_image * 255).astype(np.uint8)
    return warped_image


if __name__ == "__main__":
    import imageio

    image = imageio.v2.imread("imageio:astronaut.png")

    matrix = get_random_affine_matrix()
    image_affine_transformed = apply_affine_transform(image, matrix)

    image_perspective_transformed = apply_random_perspective(image)

    compare = np.hstack([image, image_affine_transformed, image_perspective_transformed])
    imageio.v2.imwrite("in_affine_perspective.png", compare)
