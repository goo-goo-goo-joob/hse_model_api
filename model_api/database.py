import json
from flask import g
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
    def __init__(self, path='main.db'):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        with self.conn:
            self.conn.executescript(INIT_STATEMENT)

    def close(self):
        self.conn.close()

    def create_model(self, id_: str, type_: str, params: str, binary: bytes) -> bool:
        sql = """INSERT INTO models (id, type, params, binary) VALUES (?, ?, ?, ?);"""
        try:
            with self.conn:
                self.conn.execute(sql, (id_, type_, json.dumps(params), binary))
            return True
        except sqlite3.IntegrityError:
            return False

    def delete_model(self, id_: str):
        sql = """DELETE FROM models WHERE id_ = ?;"""
        with self.conn:
            self.conn.execute(sql, (id_,))

    def get_model(self, id_: str) -> dict[str, typing.Any]:
        sql = """SELECT id, type, params, binary
FROM models
WHERE id = ?;"""
        for id_, type_, params, binary in self.conn.execute(sql, (id_,)):
            return {
                "id": id_,
                "type": type_,
                "params": params,
                "binary": binary,
            }


def get_database():
    if 'database' not in g:
        g.database = DataBase()
    return g.database
