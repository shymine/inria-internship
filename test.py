import aux_funcs as af

def main():
    metrics = [
        dict(
            test_top1_acc=[
                [25],
                [26],
                [28, 20],
                [29, 26],
                [29, 30],
                [30, 34, 30],
                [30, 36, 35],
                [30, 37, 40],
                [30, 37, 41, 45]
            ]
        ),
        dict(
            test_top1_acc=[
                [20],
                [27],
                [30, 25],
                [31, 29],
                [30, 32],
                [31, 34, 30],
                [31, 36, 37],
                [31, 39, 41],
                [31, 39, 42, 45]
            ]
        )
    ]

    af.plot_acc(metrics)

if __name__ == '__main__':
    main()
