import pandas as pd

class SharedState:
    def __init__(self):
        self.thz_bss = []
        self.rf_bss = []
        self.vehicles = []
        self.bs_performance_table = None
        self.bs_assignment_table = None

    def get_bs_performance_table(self):
        return self.bs_performance_table
