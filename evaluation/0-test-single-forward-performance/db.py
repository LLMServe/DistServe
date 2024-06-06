import sqlite3

from structs import *

class RecordManager:
    """
    To speed up our experiment, we store previous experiment results in a database.
    """
    def __init__(
        self,
        filename
    ):
        self.con = sqlite3.connect(filename)
        self.con.row_factory = sqlite3.Row
        self.cur = self.con.cursor()
        self._create_table()

    def _create_table(self):
        self.cur.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tp_world_size INTEGER,
                batch_size INTEGER,
                input_len INTEGER,
                output_len INTEGER,
                avg_prefill_time_usage REAL,
                avg_decoding_time_usage REAL,
                prefill_time_stddev REAL,
                decoding_time_stddev REAL,
                tag VARCHAR
            )
            """
        )
        self.con.commit()
    
    def _get_tag(
        self,
        worker_param: WorkerParam,
    ) -> str:
        return worker_param.model_dir
    
    def query_record(
        self,
        worker_param: WorkerParam,
        input_param: InputParam
    ) -> sqlite3.Row:
        """
        Query the record from the database. If the record does not exist, return None
        """
        tag = self._get_tag(worker_param)
        self.cur.execute(
            """
            SELECT * FROM records
            WHERE tp_world_size = ? AND
                batch_size = ? AND input_len = ? AND output_len = ? AND
                tag = ?
            """,
            (worker_param.tp_world_size,
             input_param.batch_size, input_param.input_len, input_param.output_len,
             tag)
        )
        return self.cur.fetchone()
    
    def update_or_insert_record(
        self,
        worker_param: WorkerParam,
        input_param: InputParam,
        avg_prefill_time_usage: float,
        avg_decoding_time_usage: float,
        prefill_time_stddev: float,
        decoding_time_stddev: float
    ):
        """
        Update or insert a new record
        """
        tag = self._get_tag(worker_param)
        if self.query_record(worker_param, input_param) != None:
            self.cur.execute(
                """
                UPDATE records SET 
                    avg_prefill_time_usage = ?,
                    avg_decoding_time_usage = ?,
                    prefill_time_stddev = ?,
                    decoding_time_stddev = ?
                    WHERE
                        tp_world_size = ? AND
                        batch_size = ? AND
                        input_len = ? AND
                        output_len = ? AND
                        tag = ?
                """,
                (avg_prefill_time_usage, avg_decoding_time_usage, prefill_time_stddev, decoding_time_stddev,
                 worker_param.tp_world_size,
                input_param.batch_size, input_param.input_len, input_param.output_len,
                tag)
            )
        else:
            self.cur.execute(
                """
                INSERT INTO records (tp_world_size, batch_size, input_len, output_len, avg_prefill_time_usage, avg_decoding_time_usage, prefill_time_stddev, decoding_time_stddev, tag) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (worker_param.tp_world_size,
                 input_param.batch_size, input_param.input_len, input_param.output_len,
                 avg_prefill_time_usage, avg_decoding_time_usage, prefill_time_stddev, decoding_time_stddev,
                 tag)
            )
        self.con.commit()