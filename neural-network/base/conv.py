import numpy as np

def conv():
    input_img = process_input_m()
    stride = process_input_int()
    kernel = process_input_m()
    padding = process_input_int()
    padding_mode = process_input_int()
    c, h, w = input_img.shape
    kc, kh, kw = kernel.shape
    output_h = (h + 2 * padding - kh) // stride + 1
    output_w = (w + 2 * padding - kw) // stride + 1

    output = np.zeros((output_h, output_w), int)

    if padding == 1:
        mode = "edge" if padding_mode == 1 else "constant"
        img_pad = np.pad(input_img, ((0, 0), (padding, padding), (padding, padding)), mode=mode)

    for i in range(output_h):
        for j in range(output_w):
            for k in range(c):
                output[i][j] = np.sum(img_pad[k, i * stride : i * stride + kh, j * stride : j * stride + kw] * kernel[k, :,:])

    out_list = list(output.reshape(-1))
    print(' '.join(str(i) for i in out_list))


def process_input_m():
    c, h ,w = map(int, input().strip().split())
    input_list = list(map(float, input().strip().split()))
    output = np.array(input_list).reshape((c,h,w))
    return output

def process_input_int():
    i = int(input())
    return i

if __name__ == '__main__':
    conv()