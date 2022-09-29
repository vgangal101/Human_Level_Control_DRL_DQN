class LinearSchedule():
    def __init__(self,final_exploration_frame,eps_start,eps_final):
        self.final_exploration_frame = final_exploration_frame
        self.eps_start = eps_start
        self.eps_final = eps_final 

    def get_epsilon(self, timestep):
        fraction  = min(float(timestep) / self.final_exploration_frame, 1.0)
        return self.eps_start + fraction * (self.eps_final - self.eps_start)
