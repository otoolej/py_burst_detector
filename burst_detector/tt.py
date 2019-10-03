"""
TESTING !!!

John M. O' Toole, University College Cork
Started: 02-10-2019
last update: Time-stamp: <2019-10-02 17:00:35 (otoolej)>
"""


def tt(arg1, arg2, **arg_list):
    default_args = {'freq_band': (0.5, 3),
                    'total_freq_band': (0.5, 30),
                    'DBplot': False}

    print("arg1={0}; arg2={1}".format(arg1, arg2))
    arg_list_r = {**default_args, **arg_list}
    for kw in arg_list_r:
        print(kw, ":", arg_list_r[kw])


tt(1, 2, freq_band=[0.5, 8], total_freq_band=(0.5, 30), DBplot=False)
tt(1, 2)
tt(1, 2, freq_band=(0.5, 16))
tt(1, 2, DBplot=True)


my_first_dict = {"A": 1, "B": 2}
my_second_dict = {"B": 3, "D": 4}
my_merged_dict = {**my_first_dict, **my_second_dict}

for k, v in my_merged_dict.items():
    print("key={0}; value={1}".format(k, v))
    k = v


# print(my_merged_dict)
