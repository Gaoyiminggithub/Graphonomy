class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'cihp':
            return './data/datasets/CIHP_4w/'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
