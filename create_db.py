import sqlite3

#criar banco de dados
conn=sqlite3.connect('usersdatabase.db')

c=conn.cursor()

sql="""
    DROP TABLE IF EXISTS users;
    CREATE TABLE users (
            id integer unique primary key autoincrement,
            name text);
    """
# Tabela users com duas colunas(id e nome).
c.executescript(sql)

print('[SISTEMA] Banco de dados criado com sucesso!')
conn.commit()
conn.close()
