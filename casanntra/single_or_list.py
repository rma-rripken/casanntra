from functools import wraps


def single_or_list(*arg_names):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Convert args to a dictionary for easier handling
            from inspect import signature

            sig = signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Extract the relevant arguments
            target_args = {
                name: bound_args.arguments[name]
                for name in arg_names
                if name in bound_args.arguments
            }

            # Determine if any argument is a list
            is_list = {
                name: isinstance(value, list) for name, value in target_args.items()
            }

            # If none are lists, call the function normally
            if not any(is_list.values()):
                return func(*args, **kwargs)

            # Ensure all specified arguments that are lists have the same length
            list_lengths = {
                name: len(value)
                for name, value in target_args.items()
                if isinstance(value, list)
            }

            if len(set(list_lengths.values())) > 1:
                raise ValueError(
                    f"Arguments {', '.join(list_lengths.keys())} must have the same length when they are lists."
                )

            # If lists are present, execute function element-wise
            max_length = next(iter(list_lengths.values()))  # Get the consistent length
            results = []
            for i in range(max_length):
                # Create new arguments with extracted elements
                new_kwargs = bound_args.arguments.copy()
                for name in target_args:
                    if is_list[name]:
                        new_kwargs[name] = target_args[name][i]

                results.append(func(**new_kwargs))

            return results

        return wrapper

    return decorator
