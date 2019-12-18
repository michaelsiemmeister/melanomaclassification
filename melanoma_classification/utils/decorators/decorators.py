
import copy


def decorator_deepcopy_arguments_and_return_value(f):
    def f_wrapper(*args, **kwargs):
        # deepcopy the arguments and keyword arguments
        (copied_args, copied_kwargs) = tuple(
            map(copy.deepcopy, (args, kwargs)))
        # call the function
        return_value = f(*copied_args, **copied_kwargs)

        # deepcopy the return values
        copied_return_value = copy.deepcopy(return_value)
        return copied_return_value

    return f_wrapper
