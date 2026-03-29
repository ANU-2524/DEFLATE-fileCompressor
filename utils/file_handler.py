class FileHandler :
    @staticmethod
    def read_file(path) :
        with open(path , 'r' , encoding='utf-8') as f :
            return f.read()

    @staticmethod 
    def write_file(path , data) :
        with open(path , 'w' , encoding='utf-8' ) as f :
            f.write(data)
    
    @staticmethod
    def write_binary(path , data) :
        with open(path , 'wb' ) as f :
            f.write(data) 
    
    @staticmethod
    def read_binary(path) :
        with open(path , 'r') as f :
            return f.read()