import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# config
cred = credentials.Certificate("./static/pdiot-c-firebase-adminsdk-z58yq-6a5323f199.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://pdiot-c-default-rtdb.europe-west1.firebasedatabase.app/'
})


def test(data):
    chd = db.reference()
    chd.push(data)


def _check_user_exist(username):
    chd = db.reference()
    try:
        var = chd.get()
        if var is None:
            return 0
        var = var[username]
    except KeyError:
        return 0
    return var


def username_change(old_name, new_name):
    if not _check_user_exist(old_name):
        return 0
    if _check_user_exist(new_name):
        return 0
    chd = db.reference()
    vals = chd.get()
    for key, values in vals.items():
        if key == old_name:
            chd = db.reference('/' + new_name)
            chd.set(values)
            chd = db.reference('/' + key)
            chd.delete()


def create_account_to_db(username, password):
    if _check_user_exist(username):
        return 0
    chd = db.reference('/'+username)
    chd.set({'password': password, 'history_data': [0]*14})
    return 1


def upload_data(username, device, data_value, outcome):
    if not _check_user_exist(username):
        return 0
    chd = db.reference('/' + username+'/history_data')
    temp_data = chd.get()
    temp_data[outcome] += 1
    chd.update({outcome: temp_data[outcome]})
    chd = db.reference('/' + username + '/'+device)
    chd.push({'data_value': data_value, 'predicted_outcome': outcome})


def check_username_password(username, password):
    if not _check_user_exist(username):
        return 0
    chd = db.reference('/' + username)
    pwd = chd.get()['password']
    if pwd != password:
        return 0
    return 1


def history_classification(username):
    chd = db.reference('/' + username + '/history_data')
    return chd.get()


#
# if __name__ == '__main__':
#     print(check_username_password('aa','123123'))

    # 'respect': [123132123], 'thingy': [45646545]
    # username = 'xxx'
    # create_account_to_db(username, '123465as5')
    # for i in range(3):
    #     upload_data(username,'thingy', i,'asd')
    # username_change(username,'aa')

    # for key, values in users.items():
    #     print(key, values)
    #     if key == 'xxx2':
    #         ref = db.reference('/1231321')
    #         ref.set(values)
    #         ref = db.reference('/' + key)
    #         ref.delete()
