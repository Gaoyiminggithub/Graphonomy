class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return './data/datasets/pascal/'  # folder that contains pascal/.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
