Dependencias en requirements.txt

# Iniciar servidor

python3 manage.py runserver

# Crear cuenta admin

python3 manage.py createsuperuser

# Crear cuentas, dashboard de admin

http://127.0.0.1:8000/admin/
En el dashboard de admin se pueden crear cuentas y ver los logs

#Hacer login 

http://127.0.0.1:8000/backend/token
Pasar como campos de form: username y password
Devuelve token de ingreso
El de refresh es por si quieres hacer refresh a la sesión.

# Hacer predicción

http://127.0.0.1:8000/backend/upload-and-predict/
Pasar archivo como 'file'
Pasar token de autenticación como Authorization 

