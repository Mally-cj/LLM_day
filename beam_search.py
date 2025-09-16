import numpy as np

class BeamSearch:
    def __init__(self, model, beam_width=5):
        """
        beam_width: 搜索宽度
        """
        self.model = model
        self.beam_width = beam_width
    
    def predict_next_word(self, sequence):
        """
        使用模型和当前序列预测下一个词的概率分布
        """
        # 这里假设model.predict_next_word返回的是一个概率分布
        return self.model.predict_next_word(sequence)
    
    def search(self, start_sequence, max_length=20):
        """
        执行Beam Search搜索
        start_sequence: 起始序列 (list of int)
        max_length: 最大生成长度
        
        Return:
        best_sequence: 最优序列 (list of int)
        best_score: 最优序列的得分 (float)
        """
        # 初始化
        beam = [(start_sequence, 0.0)]
        
        for _ in range(max_length - len(start_sequence)):
            new_beam = []
            
            for sequence, score in beam:
                next_probs = self.predict_next_word(sequence) # 获取当前序列的下一个词的概率分布
                # 对概率分布进行排序并获取top-k个词汇
                top_indices = np.argsort(next_probs)[-self.beam_width:]
                top_scores = next_probs[top_indices]
                
                for index, prob in zip(top_indices, top_scores):
                    new_sequence = sequence + [index]
                    new_score = score + np.log(prob)
                    new_beam.append((new_sequence, new_score))
            
            # 对新的beam按分数排序并截取前beam_width个
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = new_beam[:self.beam_width]
        
        # 返回得分最高的序列
        best_sequence, best_score = max(beam, key=lambda x: x[1])
        return best_sequence, best_score


class LanguageModel:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    
    def predict_next_word(self, sequence):
        # 随机生成概率分布
        probs = np.random.dirichlet(np.ones(self.vocab_size), size=1).flatten()
        return probs


if __name__ == "__main__":
    vocab_size = 10
    model = LanguageModel(vocab_size)
    beam_search = BeamSearch(model, beam_width=3)

    start_sequence = [1]
    best_sequence, best_score = beam_search.search(start_sequence, max_length=10)
    
    print(f"Best Sequence: {best_sequence}")
    print(f"Best Score: {best_score:.4f}")
