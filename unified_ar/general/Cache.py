import hashlib
import logging
logger = logging.getLogger(__file__)
GlobalDisable=True
def hashkey(key):
   hash_object = hashlib.sha1(key.encode('utf-8'))
   return  hash_object.hexdigest()


cachefolder="cache"
import general.utils as utils
import os.path
def get(key,valf):
   hkey=hashkey(key)
   try:
      val=utils.loadState(cachefolder,hkey)
      logger.debug(f'cached file found {key} {hkey}')
   except Exception as e:
      
      if not(os.path.exists(f'save_data/{hkey}')):
        logger.debug(f'cached file not found {key} {hkey}')
      else:
          logger.error(f'error in cached {e}',exc_info=True)
      
      val=valf()
      utils.saveState(val,cachefolder,hkey)
      with open(f'save_data/{cachefolder}/{hkey}.txt', 'w') as f:
         print(key,file=f)
         f.close()

      
   return val


def removeCache():
   import shutil
   shutil.rmtree('save_data/'+cachefolder)

import functools
def cachefunc(_func=None,*,key='',params2key=True,disable=False):
   
   def wrapper2(func):
      @functools.wraps(func)
      def wrapper(*args, **kwargs):
         
         if(('cache' in kwargs and kwargs.pop('cache')==False) or GlobalDisable or disable ):
            return func(*args, **kwargs)
         else:
            return get(str(*args)+str(**kwargs)+key,lambda:func(*args, **kwargs))    

      return wrapper
   if(_func is None):
      return wrapper2
   else:
      return wrapper2(_func)


@cachefunc
def test(i):
   print(i)
   return i*i


if __name__ == '__main__':
   logger.debug=print
   logger.debug('2')
   
   test(1)