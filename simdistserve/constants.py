class ModelTypes:
    opt_13b = 'OPT-13B'
    opt_66b = 'OPT-66B'
    opt_175b = 'OPT-175B'

    @staticmethod
    def formalize_model_name(x):
        if x == 'opt13b':
            return ModelTypes.opt_13b
        if x == 'opt66b':
            return ModelTypes.opt_66b
        if x == 'opt175b':
            return ModelTypes.opt_175b
        raise ValueError(x)
