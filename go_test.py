# from testing_functions import compare_matlab_rand_sign
from testing_functions import compare_matlab_rand_sign
# from testing_functions import compare_py_transplant

# compare_py_transplant.compare_test_fns(10, False)

# compare_matlab_rand_sign.compare_test_r2(15, False)
# compare_matlab_rand_sign.compare_generic_feat('psd_r2', 15, None, True)
# compare_matlab_rand_sign.compare_generic_feat(
#     'rel_spectral_power', 10, None, True)




# -------------------------------------------------------------------
#  compare functions
# -------------------------------------------------------------------
me = compare_matlab_rand_sign.open_matlab_engine()
# compare_matlab_rand_sign.compare_generic_feat(mat_eng=me, feat_type='rel_spectral_power',
#                                               Niter=20, params=None, DBplot=False)
# compare_matlab_rand_sign.compare_generic_feat(mat_eng=me, feat_type='psd_r2',
#                                               Niter=20, params=None, DBplot=False)
compare_matlab_rand_sign.compare_generic_feat(mat_eng=me, feat_type='envelope',
                                              Niter=20, params=None, DBplot=False)
