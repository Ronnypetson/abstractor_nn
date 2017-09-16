class AbGraph:
    def __init__(self,input_len,output_len):
        self.id_count = input_len+output_len
        self.input_len = input_len
        self.output_len = output_len
        self.nodes = set() # Nodes identification
        self.front = set()  # Nodes that can be removed
        self.abstractors = {}   # Only one abstractor per pair of nodes
        for i in range(self.id_count):
            self.nodes.add(i)   # 0 to input_len-1, input_len to input_len+output_len-1
            self.front.add(i)

    def insert_ab(self,ab_in):   # ab_in must be tuple of shape (2)
        for i in ab_in:
            if i not in self.nodes:
                return False
        self.nodes.add(self.id_count)
        self.abstractors[ab_in] = self.id_count
        for i in ab_in:
            if i in self.front:
                self.front.remove(i)
        self.front.add(self.id_count)
        self.id_count += 1
        return True

    def delete_ab(self,ab_in):    # Can only delete from the front
        y = self.abstractors[ab_in]
        if ab_in in self.abstractors and y in self.front:
            del self.abstractors[ab_in]
            self.front.remove(y)
            self.nodes.remove(y)
            return True
        return False

