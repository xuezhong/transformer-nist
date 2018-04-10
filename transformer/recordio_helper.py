import paddle.fluid as fluid


class FieldHelper(object):
    def __init__(self, filenames):
        self.fields = []
        self.shapes = []
        self.dtypes = []
        self.filenames = filenames

    def append_field(self, field_name, shape, dtype):
        self.fields.append(field_name)
        self.shapes.append(shape)
        self.dtypes.append(dtype)

    def create_reader(self):
        file_obj = fluid.layers.open_recordio_file(
            filename=self.filenames[0],
            dtypes=self.dtypes,
            shapes=self.shapes,
            lod_levels=[0] * len(self.shapes)
        )

        vars = fluid.layers.read_file(file_obj)

        result = dict()

        for var_name, var in zip(self.fields, vars):
            result[var_name] = var

        return result
