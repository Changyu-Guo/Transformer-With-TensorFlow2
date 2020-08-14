# -*- coding: utf - 8 -*-

import os

path = ''

# 删除文件，只能删除文件
# 删除目录回返回错误
# os.unlink(path)


# 返回一个 generator
# 每次 generate 一个三元组
# (dir_path, [dir_names], [file_names])
gen = os.walk('E:\\')
for item in gen:
    print(item)
