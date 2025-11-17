## 两个字符串，计算最长公共子序列


# 向下分数组
# 分到底，比较第一个，
# O(1),swap交换，如果右边

def quick_sort(nums,p_lef,p_rig):
    
    if p_lef == p_rig:
        return [nums[p_lef]]
    
    p_mid= (p_lef+p_rig)//2
    lef=quick_sort(nums,p_lef,p_mid)
    rig=quick_sort(nums,p_mid+1,p_rig)


    #对于nums排序，如果左边小于
    


    ans=[]
    q1=p_lef
    q2=p_rig
    while q1 <= p_mid and q2 <= p_mid:
        if lef[q1] < rig[q2]:
            ans.append(lef[q1])
            q1+=1
        else:
            ans.append(rig[q2])
            q2+=1
    
    print(f"lef={lef}, q1={q1},p_mid={p_mid}")
    print(f"rig={rig}, q2={q2},p_rig={p_rig}")

    
    if q1 <=p_mid:     ##如果还有剩余，就加入ans
        for i in range(q1,p_mid+1):
            ans.append(lef[i])


    if q2<=p_rig:
        for i in range(q2,p_rig+1):
            ans.append(rig[i])
    
    return ans


nums=[1,4,9,3,5,8,9]
ans=quick_sort(nums,0,len(nums)-1)
print(ans)




## 