import copy

from decorators import decorator_deepcopy_arguments_and_return_value


def dummy_function(l, kw_l=[]):
    ret_val = l + kw_l
    tuple(map(print, (l, kw_l, ret_val)))
    tuple(map(print, map(id, (l, kw_l, ret_val))))
    return (l, kw_l, ret_val)


decorated_dummy_function = (
    decorator_deepcopy_arguments_and_return_value(dummy_function))


def test_deepcopy_decorator():
    arg_list = [3, 4]
    kw_list = [5, 6, 7]
    val1 = dummy_function(arg_list, kw_l=kw_list)
    print(val1)

    val2 = decorated_dummy_function(arg_list, kw_l=kw_list)
    print(val2)

    print('\n', val1, val2, sep='\n')
    assert all(id(x) != id(y) for x, y in zip(val1, val2))
