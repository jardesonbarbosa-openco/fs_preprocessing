from abc import ABC, abstractmethod
from enum import Enum
from typing import IO, Any, Callable, Union
import os
import pandas as pd

class DataSource(str, Enum):
    API = 'api'
    BIGQUERY = 'bigquery'
    FILE = 'file'
    POSTGRES = 'postgres'
    REDSHIFT = 'redshift'
    S3 = 's3'
    SNOWFLAKE = 'snowflake'

class FileFormat(str, Enum):
    CSV = 'csv'
    JSON = 'json'
    PARQUET = 'parquet'

class ExportWritePolicy(str, Enum):
    APPEND = 'append'
    FAIL = 'fail'
    REPLACE = 'replace'

class BaseIO(ABC):
    """
    Data loader interface. All data loaders must inherit from this interface.
    """

    @abstractmethod
    def load(self, *args, **kwargs) -> pd.DataFrame:
        """
        Loads a data frame from source, returns it to memory.
        Returns:
            DataFrame: dataframe returned by the source.
        """
        pass

    @abstractmethod
    def export(self, df: pd.DataFrame, *args, **kwargs) -> None:
        """
        Exports the input dataframe to the specified source.
        Args:
            df (DataFrame): Data frame to export.
        """
        pass

class BaseFile(BaseIO):
    """
    Data loader for file-like data sources (for example, loading from local
    filesystem or external file storages such as AWS S3)
    """

    def _get_file_format(self, filepath):
        return os.path.splitext(os.path.basename(filepath))[-1][1:]
    
    def __get_reader(self, format: Union[FileFormat, str]) -> Callable:
        """
        Gets data frame reader based on file format
        Args:
            format (Union[FileFormat, str]): Format to get reader for.
        Raises:
            ValueError: Raised if invalid format specified.
        Returns:
            Callable: Returns the reader function that reads a dataframe from file
        """
        if format == FileFormat.CSV:
            return pd.read_csv
        elif format == FileFormat.JSON:
            return pd.read_json
        elif format == FileFormat.PARQUET:
            return pd.read_parquet
        else:
            raise ValueError(f'Invalid format \'{format}\' specified.')
    
    def _read(
        self,
        input: Union[IO, os.PathLike],
        format: Union[FileFormat, str],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Loads the data frame from the filepath or buffer specified.
        Args:
            input (Union[IO, os.PathLike]): Input buffer to read dataframe from.
            Can be a stream or a filepath.
            format (Union[FileFormat, str]): Format of the data frame as stored
            in stream or filepath.
        Returns:
            DataFrame: Data frame object loaded from the specified data frame.
        """
        reader = self.__get_reader(format)
        df = reader(input, **kwargs)
        return df
    
    def __get_writer(
        self,
        df: pd.DataFrame,
        format: Union[FileFormat, str],
    ) -> Callable:
        """
        Fetches the appropriate file writer based on format
        Args:
            df (DataFrame): Data frame to get file writer for.
            format (Union[FileFormat, str]): Format to write the data frame as.
        Returns:
            Callable: File writer method
        """
        if format == FileFormat.CSV:
            return df.to_csv
        elif format == FileFormat.JSON:
            return df.to_json
        elif format == FileFormat.PARQUET:
            return df.to_parquet
        else:
            raise ValueError(f'Unexpected format provided: {self.format}')
    
    def _write(
        self,
        df: pd.DataFrame,
        format: Union[FileFormat, str],
        output: Union[IO, os.PathLike],
        **kwargs,
    ) -> None:
        """
        Base method for writing a data frame to some buffer or file.
        Args:
            df (DataFrame): Data frame to write.
            format (Union[FileFormat, str]): Format to write the data frame as.
            output (Union[IO, os.PathLike]): Output stream/filepath to write data frame to.
        """
        writer = self.__get_writer(df, format)
        writer(output, **kwargs)

class BaseSQLDatabase(BaseIO):
    """
    Base data loader for connecting to a SQL database. This adds 'query' method which allows a user
    to send queries to the database server.
    """

    @abstractmethod
    def execute(self, query_string: str, **kwargs) -> None:
        """
        Sends query to the connected database
        Args:
            query_string (str): Query to send to the connected database.
            **kwargs: Additional arguments to pass to query, such as query configurations
        """
        pass

    def sample(self, schema: str, table: str, size: int = 10000, **kwargs) -> pd.DataFrame:
        """
        Sample data from a table in the connected database. Sample is not
        guaranteed to be random.
        Args:
            schema (str): The schema to select the table from.
            size (int): The number of rows to sample. Defaults to 10,000
            table (str): The table to sample from in the connected database.
        Returns:
            DataFrame: Sampled data from the data frame.
        """
        return self.load(f'SELECT * FROM {schema}.{table} LIMIT {str(size)};', **kwargs)

    def _clean_query(self, query_string: str) -> str:
        """
        Cleans query before sending to database. Cleaning steps include:
        - Removing surrounding whitespace, newlines, and tabs
        Args:
            query_string (str): Query string to clean
        Returns:
            str: Clean query string
        """
        return query_string.strip(' \n\t')


class BaseSQLConnection(BaseSQLDatabase):
    """
    Data loader for connected SQL data sources. Can be used as a context manager or by manually opening or closing the connection
    to the SQL data source after data loading is complete.
    """

    def __init__(self, verbose=False, **kwargs) -> None:
        """
        Initializes the connection with the settings given as keyword arguments. Specific data loaders will have access to different settings.
        """
        super().__init__(verbose=verbose)
        self.settings = kwargs
    
    def close(self) -> None:
        """
        Close the underlying connection to the SQL data source if open. Else will do nothing.
        """
        if '_ctx' in self.__dict__:
            self._ctx.close()
            del self._ctx
        if self.verbose and self.printer.exists_previous_message:
            print('')
    
    def commit(self) -> None:
        """
        Commits all changes made to database since last commit
        """
        self.conn.commit()
    
    @property
    def conn(self) -> Any:
        """
        Returns the connection object to the SQL data source. The exact connection type depends
        on the source and the definition of the data loader.
        """
        try:
            return self._ctx
        except AttributeError:
            raise ConnectionError(
                'No connection currently open. Open a new connection to access this property.'
            )

    @abstractmethod
    def open(self) -> None:
        """
        Opens an underlying connection to the SQL data source.
        """
        pass

    def rollback(self) -> None:
        """
        Rolls back (deletes) all changes made to database since last commit.
        """
        self.conn.rollback()

    def __del__(self):
        self.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()