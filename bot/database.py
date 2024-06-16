import mysql.connector
from mysql.connector import Error

from utils import is_running_in_docker, getenv_or_throw_exception, get_docker_secret

HOST = 'MYSQL_HOST'
USER = 'MYSQL_USER'
PASSWORD = 'MYSQL_PASSWORD'
DATABASE = 'MYSQL_DATABASE'


def create_connection():
    connection = None
    try:
        connection = mysql.connector.connect(
            host=get_docker_secret(HOST) if is_running_in_docker() else getenv_or_throw_exception(HOST),
            user=get_docker_secret(USER) if is_running_in_docker() else getenv_or_throw_exception(USER),
            password=get_docker_secret(PASSWORD) if is_running_in_docker() else getenv_or_throw_exception(PASSWORD),
            database=get_docker_secret(DATABASE) if is_running_in_docker() else getenv_or_throw_exception(DATABASE)
        )
    except Error as e:
        print(f'The error "{e}" occurred while connecting to database')
    return connection


def save_rating(user_id, rating):
    connection = create_connection()
    cursor = connection.cursor()
    query = '''
    INSERT INTO ratings (user_id, rating)
    VALUES (%s, %s)
    ON DUPLICATE KEY UPDATE rating = VALUES(rating)
    '''
    cursor.execute(query, (user_id, rating))
    connection.commit()
    connection.close()


def get_ratings():
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT rating FROM ratings')
    rows = cursor.fetchall()
    ratings = [row[0] for row in rows]
    connection.close()
    return ratings


def check_rating(user_id):
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT rating FROM ratings WHERE user_id = %s', (user_id,))
    row = cursor.fetchone()
    connection.close()
    return row[0] if row else None


def delete_rating(user_id):
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute('DELETE FROM ratings')
    connection.commit()
    connection.close()
