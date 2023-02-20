import SystematicBackdoor


def main():
    """
    Main function
    """
    SystematicBackdoor.SystematicBackdoor(
        model='resnet18',
        dataset='cifar10',
        attack='fgsm',
        defense='none'
    )


if __name__ == '__main__':
    main()
