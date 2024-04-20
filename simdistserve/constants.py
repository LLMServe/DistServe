class ModelTypes:
    opt_13b = 'OPT-13B'
    opt_66b = 'OPT-66B'
    opt_175b = 'OPT-175B'

    @staticmethod
    def formalize_model_name(x):
        if x == ModelTypes.opt_13b:
            return 'facebook/opt-13b'
        if x == ModelTypes.opt_66b:
            return 'facebook/opt-66b'
        if x == ModelTypes.opt_175b:
            return 'facebook/opt-175b'
        raise ValueError(x)
