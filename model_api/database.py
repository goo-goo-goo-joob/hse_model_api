import json
import typing
import psycopg2

INIT_STATEMENT = """CREATE TABLE IF NOT EXISTS models
(
    id  varchar(64),
    type       varchar(64),
    params     text,
    model_binary    bytea,
    PRIMARY KEY (id)
);"""


class DataBase:
    """
    Class for database operations
    """

    def __init__(self, dsn):
        self.conn = psycopg2.connect(dsn)
        with self.conn.cursor() as cursor:
            cursor.execute(INIT_STATEMENT)

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
        sql = """INSERT INTO models (id, type, params, model_binary) VALUES (%s, %s, %s, %s);"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql, (id_, type_, json.dumps(params), binary))
            return True
        except psycopg2.IntegrityError:
            return False

    def delete_model(self, id_: str):
        """
        Delete model from database
        :param id_: unique string identifier
        :return:
        """
        sql = """DELETE FROM models WHERE id = %s;"""
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (id_,))

    def get_model(self, id_src: str) -> dict[str, typing.Any]:
        """
        Get model info from database
        :param id_src: unique string identifier
        :return: model info
        """
        sql = """SELECT id, type, params, model_binary
                 FROM models
                 WHERE id = %s;"""
        with self.conn.cursor() as cursor:
            cursor.execute(sql, (id_src,))
            for id_, type_, params, binary in cursor.fetchall():
                # assert type(binary) == bytes, "Wrong type: {}".format(type(binary))
                return {
                    "id": id_,
                    "type": type_,
                    "params": json.loads(params),
                    "binary": binary.tobytes(),
                }

    def get_models(self) -> list[dict]:
        """
        Get info about all fitted models
        :return: models info
        """
        sql = """SELECT id, type, params FROM models"""
        result = []

        with self.conn.cursor() as cursor:
            cursor.execute(sql)
            for id_, type_, params in cursor.fetchall():
                result.append(
                    {
                        "id": id_,
                        "type": type_,
                        "params": json.loads(params),
                    }
                )
        return result
