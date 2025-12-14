# IoT

## 논문 원본 깃허브 링크  
https://github.com/litian96/ditto?tab=readme-ov-file

## 재구현 과정  
Anaconda로 재구현 시도.  
원본에서는 .sh로 되어 있어 유닉스, 리눅스 기반으로 진행했다는 것을 확인.  
바로 실행하려 했으나 tensorflow 버전을 낮춰야 한다는 오류 발생.  
원본에서 요구하는 라이브러리 설치 과정에서 tensorflow-gpu=1.10 설치 실패.  
파이썬 버전 3.6으로 재설치.  
원본 깃허브에 게제된 데이터셋 링크를 통해 celeba와 femnist 데이터셋 다운로드.  
femnist로 진행해보니 오래걸리기에 가장 빠르게 해볼 수 있는 데이터셋인 Vehicle의 데이터셋을 구할 방법을 탐색.  
검색 끝에 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html 이 링크에 있음을 확인.  
다운로드 후 원본에 데이터 전처리 파일인 ditto-master/data/vehicle/create_dataset.py 를 통해 학습 가능한 형태로 전환하려고 했으나 실패.  
get_real_vehicle.py 파일을 생성하여 이 코드로 다운로드 받은 vehicle 데이터셋을 먼저 전처리 후에 create_dataset.py를 진행.  
최종적으로 실행해보고 재구현 완료.   

### 실행 코드  
python main.py --dataset vehicle --model svm --optimizer ditto --learning_rate 0.1 --num_rounds 2000 --eval_every 1 --clients_per_round 5 --batch_size 32 --q 0 --seed 0 --sampling 2 --num_corrupted 11 --lam 1 --local_iters 2 > result_vehicle_attack_ditto_2000.txt  

python main.py --dataset vehicle --model svm --optimizer fedavg --learning_rate 0.1 --num_rounds 2000 --eval_every 1 --clients_per_round 5 --batch_size 32 --q 0 --seed 0 --sampling 2 --num_corrupted 11 --local_iters 2 > result_vehicle_attack_fedavg_2000.txt  



## 수정된 부분  

### ditto-master/main.py 코드 추가  
import sys  
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  



### Vehicle dataset 다운로드 링크  
https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html  

### ditto-master/data/vehicle/create_dataseet.py 코드 추가 및 매서드 부분 수정  
    
    import os


    def preprocess(x):  
        # Convert list to numpy array  
        x = np.array(x)  
        means = np.mean(x, axis=0)  
        std = np.std(x, axis=0)  
        # Prevent division by zero  
        std[std == 0] = 1  
        x = (x - means) * 1.0 / std  
        where_are_NaNs = isnan(x)  
        x[where_are_NaNs] = 0  
        return x  


    def generate_data():  
        X = []  
        y = []  
        # Load vehicle.mat file  
        mat = scipy.io.loadmat('./raw_data/vehicle.mat')  
        raw_x, raw_y = mat['X'], mat['Y']  
        print("number of users:", len(raw_x), len(raw_y))  
    
        for i in range(NUM_USER):
            # 1. Process Input Data (X)
            user_x = raw_x[i][0]
            # Transpose if needed (Features vs Samples)
            if user_x.shape[0] < user_x.shape[1] and user_x.shape[0] < 200: 
                user_x = user_x.T
    
            print("{}-th user has {} samples".format(i, user_x.shape[0]))
            X.append(preprocess(user_x).tolist())

            # 2. Process Labels (y) [CRITICAL FIX]
            # Flatten to 1D array first to calculate stats
            flat_y = np.array(raw_y[i][0]).flatten()
            # Calculate ratio
            num = 0
            for label in flat_y:
                if label == 1: num += 1
            if len(flat_y) > 0:
                print("ratio of label 1: ", num * 1.0 / len(flat_y))

            # IMPORTANT: Reshape to (N, 1) 2D array for TensorFlow compatibility!
            # Example: [0, 1] -> [[0], [1]]
            reshaped_y = flat_y.reshape(-1, 1)
            y.append(reshaped_y.tolist())
    
        return X, y


    def main():  
        # Set paths  
        train_path = "./data/train/mytrain.json"  
        test_path = "./data/test/mytest.json"  
    
        # Create directories
        if not os.path.exists("./data/train"): os.makedirs("./data/train")
        if not os.path.exists("./data/test"): os.makedirs("./data/test")

        X, y = generate_data()

        train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
        test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
        for i in range(NUM_USER):
            uname = 'f_{0:05d}'.format(i)
        
            # Shuffle
            combined = list(zip(X[i], y[i]))
            random.seed(666 + i)
            random.shuffle(combined)
            X[i][:], y[i][:] = zip(*combined)
        
            # Split Train/Test
            num_samples = len(X[i])
            train_len = int(num_samples * 0.9)
            test_len = num_samples - train_len
        
            print("User {}: train {}, test {}".format(i, train_len, test_len))
        
            train_data['users'].append(uname)
            train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
            train_data['num_samples'].append(train_len)
        
            test_data['users'].append(uname)
            test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
            test_data['num_samples'].append(test_len)
    
        print('begin to dump file...')
        with open(train_path,'w') as outfile:
            json.dump(train_data, outfile)
        with open(test_path, 'w') as outfile:
            json.dump(test_data, outfile)
        print('Done!')

    
