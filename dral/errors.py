def fail_if_len_mismatch(list_a, list_b):
    if len(list_a) != len(list_b):
        raise ValueError(
            f'Length of first list ({len(list_a)}) does not match'
            f' length of the second list ({len(list_b)})')


def fail_if_shape_mismatch(self, *args):
    it = iter(args)
    shape = next(it).shape[1:]
    if not all(arg.shape[1:] == shape for arg in args):
        raise ValueError(f"Not all args have shape: {shape}")