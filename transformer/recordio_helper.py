import paddle.fluid as fluid
from config import data_shapes


class FieldHelper(object):
    def __init__(self, filenames):
        self.fields = []
        self.dtypes = []
        self.filenames = filenames

    def append_field(self, field_name, dtype):
        print 'Append Shape', field_name, dtype
        self.fields.append(field_name)
        self.dtypes.append(dtype)

    def create_reader(self, use_open_files=False):
        shapes = []

        for field in self.fields:
            shapes.append(data_shapes[field])
        if use_open_files:
            file_obj = fluid.layers.open_files(
                filenames=[self.filenames[0]],
                dtypes=self.dtypes,
                shapes=shapes,
                thread_num=1,
                lod_levels=[0] * len(self.shapes)
            )

        else:
            file_obj = fluid.layers.open_recordio_file(
                filename=self.filenames[0],
                dtypes=self.dtypes,
                shapes=shapes,
                lod_levels=[0] * len(self.shapes)
            )

        vars = fluid.layers.read_file(file_obj)

        result = dict()

        for var_name, var in zip(self.fields, vars):
            result[var_name] = var

        return result
