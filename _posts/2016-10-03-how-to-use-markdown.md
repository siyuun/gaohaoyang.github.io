# ��ӭʹ�� Cmd Markdown �༭�Ķ���

------

������������Ҫ����ݸ���Ч�Ĺ��߼�¼˼�룬�����ʼǡ�֪ʶ���������г��صļ�ֵ���������ˣ�**Cmd Markdown** �����Ǹ����Ĵ� ���� ����Ϊ��¼˼��ͷ���֪ʶ�ṩ��רҵ�Ĺ��ߡ� ������ʹ�� Cmd Markdown��

> * ����֪ʶ��ѧϰ�ʼ�
> * �����ռǣ����ģ���������
> * ׫д���������ĸ壨����֧�֣�
> * ׫д����ѧ�����ģ�LaTeX ��ʽ֧�֣�

![cmd-markdown-logo](https://www.zybuluo.com/static/img/logo.png)

���������ڿ�������� Cmd Markdown ���߰汾����������ǰ��������ַ���أ�

### [Windows/Mac/Linux ȫƽ̨�ͻ���](https://www.zybuluo.com/cmd/)

> �뱣���˷� Cmd Markdown �Ļ�ӭ���ʹ��˵��������׫д�¸������������������Ҳ�� <i class="icon-file"></i> **���ĸ�** ����ʹ�ÿ�ݼ� `Ctrl+Alt+N`��

------

## ʲô�� Markdown

Markdown ��һ�ַ�����䡢��д�Ĵ��ı�������ԣ��û�����ʹ����Щ��Ƿ�������С������������ɼ������������ĵ���Ʃ���������Ķ�������ĵ�����ʹ�ü򵥵ķ��ű�ǲ�ͬ�ı��⣬�ָͬ�Ķ��䣬**����** ���� *б��* ĳЩ���֣��������ǣ���������

### 1. ����һ�ݴ������� [Todo �б�](https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown#13-��������-todo-�б�)

- [ ] ֧���� PDF ��ʽ�����ĸ�
- [ ] �Ľ� Cmd ��Ⱦ�㷨��ʹ�þֲ���Ⱦ���������ȾЧ��
- [x] ���� Todo �б�����
- [x] �޸� LaTex ��ʽ��Ⱦ����
- [x] ���� LaTex ��ʽ��Ź���

### 2. ��дһ�������غ㹫ʽ[^LaTeX]

$$E=mc^2$$

### 3. ����һ�δ���[^code]

```python
@requires_authorization
class SomeClass:
    pass

if __name__ == '__main__':
    # A comment
    print 'hello world'
```

### 4. ��Ч���� [����ͼ](https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown#7-����ͼ)

```flow
st=>start: Start
op=>operation: Your Operation
cond=>condition: Yes or No?
e=>end

st->op->cond
cond(yes)->e
cond(no)->op
```

### 5. ��Ч���� [����ͼ](https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown#8-����ͼ)

```seq
Alice->Bob: Hello Bob, how are you?
Note right of Bob: Bob thinks
Bob-->Alice: I am good thanks!
```

### 6. ��Ч���� [����ͼ](https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown#9-����ͼ)

```gantt
    title ��Ŀ��������
    section ��Ŀȷ��
        �������       :a1, 2016-06-22, 3d
        �����Ա���     :after a1, 5d
        ������֤       : 5d
    section ��Ŀʵʩ
        ��Ҫ���      :2016-07-05  , 5d
        ��ϸ���      :2016-07-08, 10d
        ����          :2016-07-15, 10d
        ����          :2016-07-22, 5d
    section ��������
        ����: 2d
        ����: 3d
```

### 7. ���Ʊ���

| ��Ŀ        | �۸�   |  ����  |
| --------   | -----:  | :----:  |
| �����     | \$1600 |   5     |
| �ֻ�        |   \$12   |   12   |
| ����        |    \$1    |  234  |

### 8. ����ϸ�﷨˵��

��Ҫ�鿴����ϸ���﷨˵�������Բο�����׼���� [Cmd Markdown �����﷨�ֲ�][1]�������û����Բο� [Cmd Markdown �߽��﷨�ֲ�][2] �˽����߼����ܡ�

�ܶ���֮����ͬ������ *����������* �ı༭������ֻ��ʹ�ü���רע����д�ı����ݣ��Ϳ�������ӡˢ�����Ű��ʽ��ʡȴ�ڼ��̺͹�����֮�������л����������ݺ͸�ʽ���鷳��**Markdown ����������д��ӡˢ�����Ķ�����֮���ҵ���ƽ�⡣** Ŀǰ���Ѿ���Ϊ���������ļ���������վ GitHub �� �����ʴ���վ StackOverFlow ��������д��ʽ��

---

## ʲô�� Cmd Markdown

������ʹ�úܶ๤����д Markdown������ Cmd Markdown �����������������֪�ġ���õ� Markdown ���ߡ���û��֮һ ������Ϊ�������ֵ��������������Ǻ���һ������������д������˼���֪ʶ���Լ��Ķ������м��µ�׷�����ǰѶ�����Щ����Ļ�Ӧ������ Cmd Markdown������һ�Σ����Σ����Σ����������ε�����������ߵ����飬���ս����ݻ���һ�� **�༭/����/�Ķ�** Markdown ������ƽ̨�������������κεط����κ�ϵͳ/�豸�Ϲ�����������֡�

### 1. ʵʱͬ��Ԥ��

���ǽ� Cmd Markdown ��������һ��Ϊ�������Ϊ**�༭��**���ұ�Ϊ**Ԥ����**���ڱ༭���Ĳ�����ʵʱ����Ⱦ��Ԥ��������鿴���յİ���Ч�������������������һ�����϶���������������һ��������㷨����һ�����Ĺ�����ͬ�����ȼ۵�λ�ã����ᣡ

### 2. �༭������

Ҳ��������һ�� Markdown �﷨�����֣�������ȫ��Ϥ��֮ǰ�������� **�༭��** �Ķ���������һ������ͼ��ʾ�Ĺ�������������ʹ������ڹ������ϵ�����ʽ�����������Ծɹ�����ʹ�ü��̱�Ǹ�ʽ�������д�������ȡ�

![tool-editor](https://www.zybuluo.com/static/img/toolbar-editor.png)

### 3. �༭ģʽ

��ȫ��������ķ�ʽ�༭���֣���� **�༭������** ���Ҳ�����찴ť���߰��� `Ctrl + M`���� Cmd Markdown �л��������ı༭ģʽ������һ�����ȼ���д�����������п��ܻ�������ĵ�Ԫ�ض��Ѿ���Ų��������ˬ��

### 4. ʵʱ���ƶ��ĸ�

Ϊ�˱������ݰ�ȫ��Cmd Markdown �Ὣ��ÿһ�λ��������ݱ������ƶˣ�ͬʱ�� **�༭������** �����Ҳ���ʾ `�ѱ���` �����������赣�����������������������ߵ��𣬺�Х�����ڱ༭�Ĺ�������ʱ�ر���������߻�������һ�λص� Cmd Markdown ��ʱ�����д����

### 5. ����ģʽ

�����绷�����ȶ�������¼�¼����һ���ܰ�ȫ������д����ʱ���������ͻȻʧȥ�������ӣ�Cmd Markdown �������л�������ģʽ������������������ֱ����ڱ��أ�ֱ������ָ��ٽ����Ǵ������ƶˣ���ʹ������ָ�ǰ�ر���������ߵ��ԣ�һ��û�����⣬�ȵ��´ο��� Cmd Markdown ��ʱ�����������������߱�������ִ������ƶˡ������֮�����Ǿ�����Ŭ�����������ֵİ�ȫ��

### 6. ����������

Ϊ�˱��ڹ��������ĸ壬�� **Ԥ����** �Ķ���������������ʾ�� **����������**��

![tool-manager](https://www.zybuluo.com/static/img/toolbar-manager.jpg)

ͨ���������������ԣ�

<i class="icon-share"></i> ����������ǰ���ĸ����ɹ̶����ӣ��������Ϸ���������
<i class="icon-file"></i> �½�����ʼ׫дһƪ�µ��ĸ�
<i class="icon-trash"></i> ɾ����ɾ����ǰ���ĸ�
<i class="icon-cloud"></i> ����������ǰ���ĸ�ת��Ϊ Markdown �ı����� Html ��ʽ��������������
<i class="icon-reorder"></i> �б������������͹������ĸ嶼����������鿴������
<i class="icon-pencil"></i> ģʽ���л� ��ͨ/Vim/Emacs �༭ģʽ

### 7. �Ķ�������

![tool-manager](https://www.zybuluo.com/static/img/toolbar-reader.jpg)

ͨ�� **Ԥ����** ���Ͻǵ� **�Ķ�������**�����Բ鿴��ǰ�ĸ��Ŀ¼����ǿ�Ķ����顣

�������ϵ����ͼ������Ϊ��

<i class="icon-list"></i> Ŀ¼�����ٵ�����ǰ�ĸ��Ŀ¼�ṹ����ת������Ȥ�Ķ���
<i class="icon-chevron-sign-left"></i> ��ͼ��������߱༭�����ұ�Ԥ������λ��
<i class="icon-adjust"></i> ���⣺�����˺ڰ�����ģʽ�����⣬���� **��ɫ����**�����ţ�
<i class="icon-desktop"></i> �Ķ�������������Ķ�ģʽ�ṩ��һ�����Ķ�����
<i class="icon-fullscreen"></i> ȫ������࣬��࣬�ټ�࣬һ����ȫ����ʽ��д�����Ķ�����

### 8. �Ķ�ģʽ

�� **�Ķ�������** ��� <i class="icon-desktop"></i> ���߰��� `Ctrl+Alt+M` �漴����������Ķ�ģʽ���棬�����ڰ�����Ⱦ�ϵ�ÿһ��ϸ�ڣ����壬�ֺţ��м�࣬ǰ����ɫ����ע�˴�����ʱ�䣬Ŭ�������Ķ��������Ʒ�ʡ�

### 9. ��ǩ�����������

�ڱ༭����������λ���������¸�ʽ�����ֿ��Ա�ǩ��ǰ�ĵ���

��ǩ�� δ����

��ǩ�Ժ���ĸ��ڡ��ļ��б�����Ctrl+Alt+F����ᰴ�ձ�ǩ���࣬�û�����ͬʱʹ�ü��̻����������鿴�������ڡ��ļ��б����������ı�������������ؼ��ֹ����ĸ壬����ͼ��ʾ��

![file-list](https://www.zybuluo.com/static/img/file-list.png)

### 10. �ĸ巢���ͷ���

����ʹ�� Cmd Markdown ��¼���������������Ķ��ĸ��ͬʱ�����ǲ���ϣ������һ�������Ĺ��ߣ���ϣ������˼���֪ʶͨ�����ƽ̨����ͬ���ʵ��Ķ����飬�����Ƿ���������ͬ־Ȥ���ˣ�������������������������¼�������ǵ�˼���֪ʶ�����Ե�� <i class="icon-share"></i> (Ctrl+Alt+P) ��������ĵ������Ѱɣ�

------

��һ�θ�л������ʱ���Ķ���ݻ�ӭ�壬��� <i class="icon-file"></i> (Ctrl+Alt+N) ��ʼ׫д�µ��ĸ�ɣ�ף���������¼���Ķ���������죡

���� [@ghosert][3]     
2016 �� 07�� 07��    

[^LaTeX]: ֧�� **LaTeX** �༭��ʾ֧�֣����磺$\sum_{i=1}^n a_i=0$�� ���� [MathJax][4] �ο�����ʹ�÷�����

[^code]: �����������֧�ְ��� Java, Python, JavaScript ���ڵģ�**��ʮһ**������������ԡ�

[1]: https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown
[2]: https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown#cmd-markdown-�߽��﷨�ֲ�
[3]: http://weibo.com/ghosert
[4]: http://meta.math.stackexchange.com/questions/5020/mathjax-basic-tutorial-and-quick-reference
