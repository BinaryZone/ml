import numpy as np


def pooling():
    x, y, stride, pool_size, p = map(int, input().strip().split())
    img = []
    for i in range(x):
        img.append(list(map(int, input().strip().split())))

    img = np.array(img)

    def pooling_fuc(img, x, y, stride, pool_size, p):
        output_x = int((x - pool_size) / stride) + 1
        output_y = int((y - pool_size) / stride) + 1

        for i in range(output_x):
            for j in range(output_y):
                start_i = i * stride
                start_j = j * stride
                end_i = start_i + pool_size
                end_j = start_j + pool_size

                if p == 1:
                    out = np.average(img[start_i:end_i, start_j:end_j])
                else:
                    out = np.max(img[start_i:end_i, start_j:end_j])
                if j == output_y - 1:
                    print(str(int(out)))
                else:
                    print(str(int(out)), end=' ')

    pooling_fuc(img, x, y, stride, pool_size, p)


if __name__ == '__main__':
    pooling()
