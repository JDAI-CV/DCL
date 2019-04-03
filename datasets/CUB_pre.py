import shutil
import os

train_test_set_file = open('./CUB_200_2011/train_test_split.txt')
train_list = []
test_list = []
for line in train_test_set_file:
    tmp = line.strip().split()
    if tmp[1] == '1':
        train_list.append(tmp[0])
    else:
        test_list.append(tmp[0])
# print(len(train_list))
# print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
# print(len(test_list))
train_test_set_file.close()

images_file = open('./CUB_200_2011/images.txt')
images_dict = {}
for line in images_file:
    tmp = line.strip().split()
    images_dict[tmp[0]] = tmp[1]
# print(images_dict)
images_file.close()

# prepare for train subset
for image_id in train_list:
    read_path = './CUB_200_2011/images/'
    train_write_path = './CUB_200_2011/all/'
    read_path = read_path + images_dict[image_id]
    train_write_path = train_write_path + os.path.split(images_dict[image_id])[1]
    # print(train_write_path)
    shutil.copyfile(read_path, train_write_path)

# prepare for test subset
for image_id in test_list:
    read_path = './CUB_200_2011/images/'
    test_write_path = './CUB_200_2011/all/'
    read_path = read_path + images_dict[image_id]
    test_write_path = test_write_path + os.path.split(images_dict[image_id])[1]
    # print(train_write_path)
    shutil.copyfile(read_path, test_write_path)

class_file = open('./CUB_200_2011/image_class_labels.txt')
class_dict = {}
for line in class_file:
    tmp = line.strip().split()
    class_dict[tmp[0]] = tmp[1]
class_file.close()

# create train.txt
train_file = open('./CUB_200_2011/train.txt', 'a')
for image_id in train_list:
    train_file.write(os.path.split(images_dict[image_id])[1])
    train_file.write(' ')
    train_file.write(class_dict[image_id])
    train_file.write('\n')
train_file.close()

test_file = open('./CUB_200_2011/test.txt', 'a')
for image_id in test_list:
    test_file.write(os.path.split(images_dict[image_id])[1])
    test_file.write(' ')
    test_file.write(class_dict[image_id])
    test_file.write('\n')
test_file.close()
