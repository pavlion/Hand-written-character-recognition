from torch.utils.data.sampler import Sampler


class IndicesSampler(Sampler):
    ''' Data loader will only sample specific indices '''
 
    def __init__(self, indices, shuffle=False):

        self.indices = np.array(indices)
        self.len_ = len(self.indices)
        self.shuffle = shuffle
        
    
    def __len__(self):
        return self.len_

    def __iter__(self):

        if self.shuffle:
            np.random.shuffle(self.indices) # in-place shuffle

        for index in self.indices:
            yield index
    