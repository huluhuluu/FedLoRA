"""Initialization policies are explicit and separately registered."""


class Kaiming:
    def peft_value(self):
        return True


class Gaussian:
    def peft_value(self):
        return "gaussian"
