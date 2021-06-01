from typing import TextIO


def save(save_to, data: tuple):
    if isinstance(save_to, list):
        save_to.append(data)
    elif isinstance(save_to, TextIO):
        save_to.write(','.join([str(_) for _ in data]) + '\n')
    else:
        print(
            ' '.join(
                [
                    format(_, '.6g') if isinstance(_, float) else str(_)
                    for _ in data
                ]
            )
        )
