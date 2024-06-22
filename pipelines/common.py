import luigi


class CreateDirectory(luigi.ExternalTask):
    directory = luigi.PathParameter()

    def output(self):
        return luigi.LocalTarget(self.directory)
