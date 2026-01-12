import matplotlib.pyplot as plt
import numpy as np
import os

# 畫折線圖
def hydrograph(obv, est, save_path=None, show=False):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.plot(obv, color='red', label='Observation')
    ax.plot(est, color='blue', label='Estimation')
    ax.set_title('Hydrograph')
    ax.set_xlabel('Time(hr)')
    ax.set_ylabel('Depth(cm)')
    ax.legend()
    ax.grid()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Hydrograph saved in {save_path}")

    if show:
        plt.show()

    plt.close(fig)  # ✅ 確保釋放記憶體
    return fig


# 畫45度線圖
def scatter(obv, est, save_path=None, show=False):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    minVal = min(min(obv), min(est))
    maxVal = max(max(obv), max(est))
    line_range = [minVal, maxVal]

    ax.plot(line_range, line_range, color="red", linewidth=1, linestyle='-')
    ax.scatter(obv, est, c=np.random.rand(len(obv)), cmap='rainbow', s=20)
    ax.set_title('Scatter Plot')
    ax.set_xlabel('Observation')
    ax.set_ylabel('Estimation')
    ax.grid()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Scatter plot saved in {save_path}")

    if show:
        plt.show()

    plt.close(fig)  # ✅ 確保釋放記憶體
    return fig


# 畫出所有所需圖片並存檔
def draw_all(fig_names, fig_folders, event_num, Y_train, Y_train_predict, Y_test, Y_predict):
    for fig_name in fig_names:
        for dtype in ('train', 'test'):
            fig_path = os.path.join(
                fig_folders[fig_name],
                f'RES-{dtype}_EV{event_num+1:02d}.png'
            )

            if fig_name == 'Hydrograph':
                if dtype == 'train':
                    hydrograph(Y_train, Y_train_predict, save_path=fig_path)
                else:
                    hydrograph(Y_test, Y_predict, save_path=fig_path)

            elif fig_name == 'Scatter plot':
                if dtype == 'train':
                    scatter(Y_train, Y_train_predict, save_path=fig_path)
                else:
                    scatter(Y_test, Y_predict, save_path=fig_path)


# 跨 T-step 畫折線圖
def hydrograph_T_steps(RES_test, FutureTime, HYDROGRAPH_FOLDER):
    FutureTime = int(FutureTime)
    all_tests = []
    all_predicts = []

    for i in range(len(RES_test)):
        all_tests.append(np.array(RES_test[i][0], dtype=np.float64).T.flatten().tolist())  # Observation
        all_predicts.append(np.array(RES_test[i][1], dtype=np.float64).T.flatten().tolist())  # Estimation

    for i in range(len(all_tests)):
        obv, est = all_tests[i], all_predicts[i]
        for j in range(0, len(est)-1, FutureTime):
            fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
            ax.plot(obv, color='red', label='Observation')

            if j+FutureTime > len(est)-1:
                ax.plot(list(range(j, len(est))), est[j:len(est)], color='blue', label='Estimation')
            else:
                ax.plot(list(range(j, j+FutureTime+1)), est[j:j+FutureTime+1], color='blue', label='Estimation')

            ax.set_title('Hydrograph')
            ax.set_xlabel('Time(hr)')
            ax.set_ylabel('Depth(cm)')
            ax.legend()
            ax.grid()

            folder = os.path.join(HYDROGRAPH_FOLDER, f'RES-test_EV{i+1:02d}')
            os.makedirs(folder, exist_ok=True)

            fig_path = os.path.join(folder, f'RES-test_EV{i+1:02d}-T+{j+FutureTime}.png')
            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            print(f"Hydrograph (T-step) saved in {fig_path}")
            plt.close(fig)  # ✅ 加入釋放記憶體
