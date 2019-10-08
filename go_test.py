# from testing_functions import compare_matlab_rand_sign
# from testing_functions import compare_matlab_rand_sign
from testing_functions import compare_py_transplant


# compare_matlab_rand_sign.compare_test_r2(15, False)
# compare_matlab_rand_sign.compare_generic_feat('psd_r2', 15, None, True)
# compare_matlab_rand_sign.compare_generic_feat(
#     'rel_spectral_power', 10, None, True)


compare_py_transplant.compare_test_fns(10, False)
