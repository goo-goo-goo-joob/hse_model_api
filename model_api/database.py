import json
import sqlite3
import typing

INIT_STATEMENT = """CREATE TABLE IF NOT EXISTS models
(
    id  text,
    type       text,
    params     text,
    binary    blob,
    PRIMARY KEY (id)
);"""


class DataBase:
    """
    Class for database operations
    """

    def __init__(self, path="main.db"):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        with self.conn:
            self.conn.executescript(INIT_STATEMENT)

    def close(self):
        self.conn.close()

    def create_model(self, id_: str, type_: str, params: str, binary: bytes) -> bool:
        """
        Add new model info in database
        :param id_: unique string identifier
        :param type_: type of model
        :param params: learning parameters
        :param binary: binary representation of model
        :return: success or not
        """
        sql = """INSERT INTO models (id, type, params, binary) VALUES (?, ?, ?, ?);"""
        try:
            with self.conn:
                self.conn.execute(sql, (id_, type_, json.dumps(params), binary))
            return True
        except sqlite3.IntegrityError:
            return False

    def delete_model(self, id_: str):
        """
        Delete model from database
        :param id_: unique string identifier
        :return:
        """
        sql = """DELETE FROM models WHERE id = ?;"""
        with self.conn:
            self.conn.execute(sql, (id_,))

    def get_model(self, id_src: str) -> dict[str, typing.Any]:
        """
        Get model info from database
        :param id_src: unique string identifier
        :return: model info
        """
        sql = """SELECT id, type, params, binary
                 FROM models
                 WHERE id = ?;"""
        for id_, type_, params, binary in self.conn.execute(sql, (id_src,)):
            return {
                "id": id_,
                "type": type_,
                "params": json.loads(params),
                "binary": binary,
            }

    def get_models(self) -> list[dict]:
        """
        Get info about all fitted models
        :return: models info
        """
        sql = """SELECT id, type, params FROM models"""
        result = []
        for id_, type_, params in self.conn.execute(sql):
            result.append(
                {
                    "id": id_,
                    "type": type_,
                    "params": json.loads(params),
                }
            )
        return result
