from flow.block import Block
import os


def read_line_from_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


class Flow(object):

    def __init__(self, flow_file):
        '''
        Construct the flow based on the descriptions from flow_file

        Args:
            flow_file (str): the path pointed to the text file containing the flow instructions.
        '''

        flow_instruction = read_line_from_file(flow_file)

        self.block_dict = dict()
        self.header = None

        for row in flow_instruction:
            step_block = Block(row)
            self.block_dict[step_block.name] = step_block

            if self.header is None:
                self.header = step_block
        self.connect_blocks()

    def connect_blocks(self):
        '''
        Connect blocks
        '''
        for key, value in self.block_dict.items():
            try:
                for branch_condition, branch_block in value.branch.items():
                    value.branch[branch_condition] = self.block_dict[branch_block]
            except:
                raise Exception("Error when connecting blocks in flow")

    def __str__(self):
        '''
        Return the flow information as string.
        '''
        flow_str = ''
        for key, value in self.block_dict.items():
            flow_str += value.__str__()
        return flow_str


if __name__ == '__main__':
    flow = Flow('../OpenAGI_Flow.txt')
    print(flow.__str__())
