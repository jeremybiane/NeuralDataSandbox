import os
import sys

def get_response(quest,default=None,opts=('y','n'),please=None,fmt=None):
    quest += " ("
    quest += "/".join([('['+o+']') if o==default else o for o in opts])
    quest += "): "

    if default is not None: opts = list(opts)+['']
    if please is None: please = quest
    if fmt is None: fmt = lambda x: x

    rin = raw_input(quest)
    while fmt(rin) not in opts: rin = raw_input(please)

    return default if default is not None and rin == '' else fmt(rin)


def get_user_vars():
    """
    Get variables in user namespace (ripped directly from ipython namespace
    magic code)
    """
    import IPython
    ip = IPython.get_ipython()    
    user_ns = ip.user_ns
    user_ns_hidden = ip.user_ns_hidden
    nonmatching = object()
    var_hist = [ i for i in user_ns
                 if not i.startswith('_') \
                 and (user_ns[i] is not user_ns_hidden.get(i, nonmatching)) ]
    return var_hist


def save_workspace(db):
    """
    Prompts for saving the current session during shutdown
    """
    import IPython, cPickle
    #import cPickle
    var_hist = get_user_vars()
    ip = IPython.get_ipython()
    #db = ip.db

    # collect any variables that need to be deleted from db
    keys = map(lambda x: x.split('/')[1], ip.db.keys('autorestore/*'))
    todel = set(keys).difference(ip.user_ns)
    changed = [ip.db[k] != ip.user_ns[k.split('/')[1]]
               for k in ip.db.keys('autorestore/*') if k.split('/')[1] in ip.user_ns]
    try:
        if len(var_hist) == 0 and len(todel) == 0 and not any(changed): return
        #if get_response("Save session?",'n',fmt=str.lower)=='n': return
    except KeyboardInterrupt:
        return

    # Save interactive variables (ignore unsaveable ones)
    for name in var_hist:
        if name == 'db':
            continue
        obj = ip.user_ns[name]
        try:
            db[ name ] = obj
        except cPickle.PicklingError:
            print("Could not store variable '%s'. Skipping..." % name)
            del db[ name ]

    # Remove any previously stored variables that were deleted in this session
    for k in todel:
        del db[k]


def restore_aliases(ip, db):
    staliases = db.get('stored_aliases', {})
    for k,v in staliases.items():
        #print "restore alias",k,v # dbg
        #self.alias_table[k] = v
        ip.alias_manager.define_alias(k,v)


def refresh_variables(ip, db):
    for key in db.keys('*'):
        # strip autorestore
        justkey = os.path.basename(key)
        try:
            obj = db[key]
        except KeyError:
            print("Unable to restore variable '%s', ignoring (use %%store -d to forget!)" % justkey)
            print("The error was:", sys.exc_info()[0])
        else:
            #print "restored",justkey,"=",obj #dbg
            ip.user_ns[justkey] = obj


def restore_dhist(ip, db):
    ip.user_ns['_dh'] = db.get('dhist',[])


def load_workspace(ip, db):
    refresh_variables(ip, db)
    restore_aliases(ip, db)
    restore_dhist(ip, db)
