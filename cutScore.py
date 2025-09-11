import cv2


def get_staff_lines(img, threshold=0.6):
    h, w = img.shape
    all_lines = []  # 五线谱中所有的直线对应的行数
    row_histogram = [0] * h  # 每一行的0值统计
    staff_lines = []  # 实线
    staff_lines_thicknesses = []  # 实线厚度

    # 统计五线谱中所有直线的行数
    for i in range(h):
        for j in range(w):
            if img[i, j] != 255:
                row_histogram[i] += 1
    for row in range(len(row_histogram)):
        if row_histogram[row] >= (w * threshold):
            all_lines.append(row)

    # 得到五线谱中谱线开始对应的行数、开始到结束的厚度
    it = 0
    thickness = 1
    while it < len(all_lines):
        # 谱线开始
        if thickness == 1:
            staff_lines.append(all_lines[it])
        # 厚度计算
        if it == len(all_lines) - 1:
            staff_lines_thicknesses.append(thickness)
        elif all_lines[it] + 1 == all_lines[it + 1]:
            thickness += 1
        else:
            staff_lines_thicknesses.append(thickness)
            thickness = 1
        it += 1
    return all_lines, staff_lines, staff_lines_thicknesses


def removeTitle(img, start_line, end_line):
    h, w = img.shape
    start = 0
    end = 0
    for i in range(start_line, 0, -1):
        flag = 1
        for j in range(w):
            if img[i, j] != 255:
                flag = 0
        if flag == 1:
            start = i
            break
    for i in range(end_line, h, 1):
        flag = 1
        for j in range(w):
            if img[i, j] != 255:
                flag = 0
        if flag == 1:
            end = i
            break
    return img[start:end, :]


def cut_image_into_buckets(img, staff_lines):
    cutted_images = []
    i = 0
    # 可以划分成几片
    number = len(staff_lines) // 5
    while i < number:
        start = staff_lines[i * 5]
        end = staff_lines[i * 5 + 4]
        cutted_images.append(removeTitle(img, start, end))
        i += 1
    return cutted_images


def score2part(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    all_lines, staff_lines, staff_lines_thickness = get_staff_lines(img)
    cutted_images = cut_image_into_buckets(img, staff_lines)
    return cutted_images


def singleScore(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return img
