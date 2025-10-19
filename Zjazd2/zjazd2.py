import argparse
import tensorflow as tf
import numpy as np

@tf.function
def rotate_point(x, y, angle_deg):
    angle_rad = tf.cast(angle_deg * np.pi / 180.0, tf.float32)

    # Macierz obrotu
    rotation_matrix = tf.stack([
        [tf.cos(angle_rad), -tf.sin(angle_rad)],
        [tf.sin(angle_rad), tf.cos(angle_rad)]
    ])

    # Wektor punktu
    point = tf.constant([[x], [y]], dtype=tf.float32)

    # Obrót
    rotated_point = tf.matmul(rotation_matrix, point)

    return tf.squeeze(rotated_point)


@tf.function
def solve_linear(A, b):
    A = tf.cast(A, tf.float32)
    b = tf.cast(b, tf.float32)

    if len(b.shape) == 1:
        b = tf.expand_dims(b, 1)
    x = tf.linalg.solve(A, b)

    return tf.squeeze(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-zad", type=str,
                        choices=["rotate_point", "solve_linear", "solve_linear_auto"],
                        required=True)

    parser.add_argument("-x", type=float)
    parser.add_argument("-y", type=float)
    parser.add_argument("-angle", type=float)

    parser.add_argument("-A", type=str)
    parser.add_argument("-B", type=str)

    parser.add_argument("-n", type=str)

    args = parser.parse_args()

    if args.zad == "rotate_point":
        rotate = rotate_point(args.x, args.y, args.angle)

        print(f"Punkt wejściowy: {args.x, args.y}")
        print(f"Kąt obrotu: {args.angle}")
        print(f"Punkt po obrocie: {rotate.numpy()}")

    elif args.zad == "solve_linear":
        A = np.array([[float(num) for num in row.split()] for row in args.A.split(',')])
        b = np.array([float(num) for num in args.B.split()])
        if np.linalg.det(A) == 0:
            print("Brak jednoznacznego rozwiązania")
            return
        x = solve_linear(tf.constant(A), tf.constant(b))
        print("Rozwiązanie x =", x.numpy())

    elif args.zad == "solve_linear_auto":
        nums = [float(n) for n in args.n.split(',')]
        m = len(nums)
        delta = 1 + 4 * m
        n = int((-1 + np.sqrt(delta)) / 2 + 0.5)
        if n * (n + 1) != m:
            print(f"Liczba argumentów ({m}) nie pasuje do układu")
            return
        A = np.array(nums[:n * n]).reshape((n, n))
        b = np.array(nums[n * n:])
        if np.linalg.det(A) == 0:
            print("Brak jednoznacznego rozwiązania")
            return
        x = solve_linear(tf.constant(A), tf.constant(b))
        print("Rozwiązanie x =", x.numpy())


if __name__ == "__main__":
    main()
