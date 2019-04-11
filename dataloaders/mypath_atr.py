class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'atr':
            return './data/datasets/ATR/'  # folder that contains atr/.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
