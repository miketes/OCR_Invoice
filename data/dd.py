import numpy as np

# 2-D array: 2 x 3
two_dim_matrix_one = np.array([[0,1,0], [1,0,1]])
another_two_dim_matrix_one = np.array([[0,1], [1,0],[0,1]])

# 对应元素相乘 element-wise product
element_wise = np.mat(two_dim_matrix_one) * np.mat(another_two_dim_matrix_one)
print('element wise product: %s' %(element_wise))

# 对应元素相乘 element-wise product
element_wise_2 = np.multiply(np.mat(two_dim_matrix_one), np.mat(another_two_dim_matrix_one))
print('element wise product: %s' % (element_wise_2))