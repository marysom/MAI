CREATE TABLE films (
  title TEXT,
  id BIGINT,
  country TEXT,
  box_office FLOAT,
  releas_date DATE
)
;

CREATE TABLE person (
  id BIGINT,
  fio TEXT
)
;

CREATE TABLE person2content (
  person_id BIGINT,
  film_id BIGINT,
  person_type TEXT
)
;

INSERT INTO films VALUES();

INSERT INTO person VALUES(400001, 'Somova Maria Sergeevna');
INSERT INTO person VALUES(400002, 'Silyakova Alena Sergeevna');
INSERT INTO person VALUES(400003, 'Koltakov Ivan Pavlovich');

INSERT INTO person2content VALUES();
